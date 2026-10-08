"""
model_service_v2.py — Flask /classify endpoint.

WHAT CHANGED FROM THE ORIGINAL:
  The original version of this file loaded model_binary_v2.pkl (the plain
  RandomForest from train_model.py, ~41 causal-only features) and called
  clf.predict() directly, which uses LightGBM/sklearn's built-in 0.5 cutoff.
  It never loaded model_multiclass_enhanced.pkl or any optimal_thresholds
  file, so none of the per-class threshold tuning work actually affected
  live predictions.

  This version:
    1. Loads model_binary_enhanced.pkl + model_multiclass_enhanced.pkl +
       feature_cols_enhanced.pkl (the LightGBM hybrid pipeline from
       train_model_enhanced.py — 117 features).
    2. Computes ALL 117 features causally, per-sender, message-by-message
       (enriched deltas, direction-change, rolling WINDOW=5 variance,
       rolling SEQ_WINDOW=10 mean/std/max/min, trend, autocorr) — not just
       the ~20 base features the old service computed.
    3. Loads optimal_thresholds_v2.pkl (the corrected per-class thresholds
       from fix_threshold_leak_v2.py) and applies threshold = thresholds
       [multiclass_prediction] to the binary attack probability, exactly
       matching the "final result" decision rule actually evaluated in
       train_model_enhanced.py / fix_threshold_leak_v2.py.
    4. Falls back gracefully (with a loud startup warning) to the old,
       un-corrected optimal_thresholds.pkl, or to a flat 0.5 cutoff, if the
       v2 file isn't present yet — so this file runs today even before you
       finish running fix_threshold_leak_v2.py.

NOTE ON THE "HYBRID" OVERRIDE:
  train_model_enhanced.py computes a hybrid_pred (binary flipped to attack
  when the multiclass head predicts a hard class; flipped to normal when
  the multiclass head is very confident it's Normal) as an exploratory
  step, but that hybrid_pred was never actually fed into the script's
  final reported metrics — the "FINAL RESULTS" section there recomputes
  predictions straight from bin_proba + the per-class threshold, bypassing
  hybrid_pred entirely. To match what was actually validated, this service
  does the same (threshold-only). The hybrid override is included below
  but OFF by default (USE_HYBRID_OVERRIDE = False) so you can turn it on
  deliberately and re-validate it rather than silently inheriting unvalidated
  behavior.
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from gemma_layer import explain_anomaly

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
USE_HYBRID_OVERRIDE = False   # see note above — off by default, matches what was validated
HARD_CLASSES = {2, 4, 10, 11, 17}
WINDOW = 5          # matches train_model_enhanced.py's short rolling window (var/mean)
SEQ_WINDOW = 10      # matches train_model_enhanced.py's sequence window (mean/std/max/min/trend/autocorr)

ATTACK_NAMES = {
    0: 'Normal', 1: 'Const Position', 2: 'Const Pos Offset', 3: 'Random Position', 4: 'Random Pos Offset',
    5: 'Const Speed', 6: 'Const Speed Offset', 7: 'Random Speed', 8: 'Random Speed Offset',
    9: 'Disruptive', 10: 'Data Replay', 11: 'DoS', 12: 'DoS Random',
    13: 'DoS Disruptive', 14: 'Data Replay Sybil', 15: 'Traffic Congestion Sybil',
    16: 'DoS Random Sybil', 17: 'DoS Disruptive Sybil', 18: 'Class18', 19: 'Class19'
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("1. Loading model artifacts...")
clf_binary = joblib.load('model_binary_enhanced.pkl')
clf_multiclass = joblib.load('model_multiclass_enhanced.pkl')
feature_cols = joblib.load('feature_cols_enhanced.pkl')
print(f"✅ Loaded binary + multiclass models, {len(feature_cols)} required features.")

try:
    optimal_thresholds = joblib.load('optimal_thresholds_v2.pkl')
    print(f"✅ Loaded CORRECTED per-class thresholds (optimal_thresholds_v2.pkl): {optimal_thresholds}")
except FileNotFoundError:
    try:
        optimal_thresholds = joblib.load('optimal_thresholds.pkl')
        print("⚠️  optimal_thresholds_v2.pkl not found — falling back to the OLD, buggy "
              "optimal_thresholds.pkl. Run fix_threshold_leak_v2.py and drop its output "
              "next to this file to fix this.")
        print(f"   Loaded (uncorrected) thresholds: {optimal_thresholds}")
    except FileNotFoundError:
        optimal_thresholds = {0: 0.5}
        print("⚠️  No threshold file found at all — every class will use the default 0.5 cutoff.")

# ─────────────────────────────────────────────────────────────────────────────
# PER-SENDER CAUSAL STATE
# ─────────────────────────────────────────────────────────────────────────────
# Same pattern as the original service's sender_* dicts, extended to also carry
# what's needed for the enriched + sequence features (rolling windows, previous
# delta vectors for direction-change). Resets per-sender exactly like the
# original's loop/replay guard, so replaying data.csv in a loop still behaves.

sender_msg_count = defaultdict(int)
sender_last_time = {}
sender_recent_times = defaultdict(deque)      # trailing 1s window, for msg_rate_1s

# previous delta vector per sender per prefix, for direction-change cosine calc
sender_prev_delta = defaultdict(dict)         # sender -> {'pos': (dx,dy,dz), 'spd': ..., 'acl': ..., 'hed': ...}

# short WINDOW=5 history of the 9 raw delta components, for roll_var_mean features
sender_small_window = defaultdict(lambda: deque(maxlen=WINDOW))

# SEQ_WINDOW=10 history of the 14 sequence-feature source values
SEQ_COLS = ['pos_delta_mag', 'spd_delta_mag', 'acl_delta_mag', 'hed_delta_mag',
            'pos_dir_change', 'spd_dir_change', 'acl_dir_change', 'hed_dir_change',
            'time_since_last_msg', 'msg_rate_1s', 'spd_mag_delta',
            'pos_x_delta', 'pos_y_delta', 'pos_z_delta']
TREND_COLS = ['pos_delta_mag', 'spd_delta_mag', 'time_since_last_msg', 'msg_rate_1s']
AUTOCORR_COLS = ['pos_delta_mag', 'spd_delta_mag', 'time_since_last_msg']

sender_seq_window = defaultdict(lambda: deque(maxlen=SEQ_WINDOW))


def _reset_sender(sender):
    sender_msg_count[sender] = 0
    sender_recent_times[sender].clear()
    sender_prev_delta[sender] = {}
    sender_small_window[sender].clear()
    sender_seq_window[sender].clear()


def _sample_std(vals):
    """Matches pandas .rolling().std() default (ddof=1): NaN/0 for n<2."""
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return var ** 0.5


def _sample_var(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    return sum((v - mean) ** 2 for v in vals) / (n - 1)


def _trend(vals):
    """Linear slope over the window, matching np.polyfit(..., 1)[0]; 0 if <2 points."""
    n = len(vals)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = np.asarray(vals, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _autocorr_1(vals):
    """Lag-1 autocorrelation, matching training: 0 if <3 points or near-zero variance."""
    n = len(vals)
    if n < 3:
        return 0.0
    y = np.asarray(vals, dtype=float)
    if np.std(y) <= 1e-6:
        return 0.0
    c = np.corrcoef(y[:-1], y[1:])[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _dir_change(prev_delta, cur_delta):
    """1 - cosine_similarity(prev, cur); 0 if no prior reading or either vector ~0 — matches training."""
    if prev_delta is None:
        return 0.0
    dx0, dy0, dz0 = prev_delta
    dx1, dy1, dz1 = cur_delta
    mag0 = (dx0**2 + dy0**2 + dz0**2) ** 0.5
    mag1 = (dx1**2 + dy1**2 + dz1**2) ** 0.5
    if mag0 <= 1e-6 or mag1 <= 1e-6:
        return 0.0
    cos_sim = (dx0*dx1 + dy0*dy1 + dz0*dz1) / (mag0 * mag1)
    return 1.0 - cos_sim


def compute_causal_features(data: dict) -> dict:
    """
    Computes all 117 features in feature_cols_enhanced.pkl causally, in place,
    mirroring train_model_enhanced.py's offline pipeline exactly but one
    message at a time.
    """
    sender = data.get('sender')
    send_time = float(data.get('sendTime', 0.0))

    # ── Loop/replay guard (unchanged from original) ──
    last_t = sender_last_time.get(sender)
    if last_t is not None and send_time <= last_t:
        _reset_sender(sender)
        last_t = None

    # ── msg_count_so_far / time_since_last_msg / msg_rate_1s ──
    sender_msg_count[sender] += 1
    data['msg_count_so_far'] = sender_msg_count[sender]
    data['time_since_last_msg'] = (send_time - last_t) if last_t is not None else 999.0
    sender_last_time[sender] = send_time

    dq_t = sender_recent_times[sender]
    dq_t.append(send_time)
    while dq_t and send_time - dq_t[0] > 1.0:
        dq_t.popleft()
    data['msg_rate_1s'] = len(dq_t)

    # ── base deltas (already in data from caller) + magnitudes ──
    spdx, spdy = data.get('spdx', 0.0), data.get('spdy', 0.0)
    spdx_n, spdy_n = data.get('spdx_n', 0.0), data.get('spdy_n', 0.0)
    aclx, acly = data.get('aclx', 0.0), data.get('acly', 0.0)
    data['spd_mag'] = float(np.sqrt(spdx**2 + spdy**2))
    data['acl_mag'] = float(np.sqrt(aclx**2 + acly**2))
    data['spd_mag_n'] = float(np.sqrt(spdx_n**2 + spdy_n**2))
    data['spd_mag_delta'] = data['spd_mag'] - data['spd_mag_n']

    # ── enriched delta magnitudes ──
    data['pos_delta_mag'] = float(np.sqrt(data['pos_x_delta']**2 + data['pos_y_delta']**2 + data['pos_z_delta']**2))
    data['spd_delta_mag'] = float(np.sqrt(data['spd_x_delta']**2 + data['spd_y_delta']**2 + data['spd_z_delta']**2))
    data['acl_delta_mag'] = float(np.sqrt(data['acl_x_delta']**2 + data['acl_y_delta']**2 + data['acl_z_delta']**2))
    data['hed_delta_mag'] = float(np.sqrt(data['hed_x_delta']**2 + data['hed_y_delta']**2 + data['hed_z_delta']**2))

    # ── direction-change features (needs previous message's delta vector) ──
    prev = sender_prev_delta[sender]
    for prefix in ['pos', 'spd', 'acl', 'hed']:
        cur_vec = (data[f'{prefix}_x_delta'], data[f'{prefix}_y_delta'], data[f'{prefix}_z_delta'])
        data[f'{prefix}_dir_change'] = _dir_change(prev.get(prefix), cur_vec)
        prev[prefix] = cur_vec

    # ── WINDOW=5 rolling variance-of-deltas, averaged per group (roll_var_mean) ──
    small_dq = sender_small_window[sender]
    small_dq.append({
        'pos_x': data['pos_x_delta'], 'pos_y': data['pos_y_delta'], 'pos_z': data['pos_z_delta'],
        'spd_x': data['spd_x_delta'], 'spd_y': data['spd_y_delta'], 'spd_z': data['spd_z_delta'],
        'acl_x': data['acl_x_delta'], 'acl_y': data['acl_y_delta'], 'acl_z': data['acl_z_delta'],
    })
    for group in ['pos', 'spd', 'acl']:
        vx = [d[f'{group}_x'] for d in small_dq]
        vy = [d[f'{group}_y'] for d in small_dq]
        vz = [d[f'{group}_z'] for d in small_dq]
        data[f'{group}_delta_roll_var_mean'] = float(np.mean([_sample_var(vx), _sample_var(vy), _sample_var(vz)]))

    # ── SEQ_WINDOW=10 rolling mean/std/max/min + trend + autocorr ──
    seq_vals = {c: data[c] for c in SEQ_COLS}
    seq_dq = sender_seq_window[sender]
    seq_dq.append(seq_vals)

    for col in SEQ_COLS:
        series = [row[col] for row in seq_dq]
        data[f'{col}_roll_mean_{SEQ_WINDOW}'] = float(np.mean(series))
        data[f'{col}_roll_std_{SEQ_WINDOW}'] = _sample_std(series)
        data[f'{col}_roll_max_{SEQ_WINDOW}'] = float(np.max(series))
        data[f'{col}_roll_min_{SEQ_WINDOW}'] = float(np.min(series))

    for col in TREND_COLS:
        series = [row[col] for row in seq_dq]
        data[f'{col}_trend_{SEQ_WINDOW}'] = _trend(series)

    for col in AUTOCORR_COLS:
        series = [row[col] for row in seq_dq]
        data[f'{col}_autocorr_1'] = _autocorr_1(series)

    return data


@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json or {}

        # ── Base deltas (same as original) ──
        data['pos_x_delta'] = data.get('posx', 0.0) - data.get('posx_n', 0.0)
        data['pos_y_delta'] = data.get('posy', 0.0) - data.get('posy_n', 0.0)
        data['pos_z_delta'] = data.get('posz', 0.0) - data.get('posz_n', 0.0)
        data['spd_x_delta'] = data.get('spdx', 0.0) - data.get('spdx_n', 0.0)
        data['spd_y_delta'] = data.get('spdy', 0.0) - data.get('spdy_n', 0.0)
        data['spd_z_delta'] = data.get('spdz', 0.0) - data.get('spdz_n', 0.0)
        data['acl_x_delta'] = data.get('aclx', 0.0) - data.get('aclx_n', 0.0)
        data['acl_y_delta'] = data.get('acly', 0.0) - data.get('acly_n', 0.0)
        data['acl_z_delta'] = data.get('aclz', 0.0) - data.get('aclz_n', 0.0)
        data['hed_x_delta'] = data.get('hedx', 0.0) - data.get('hedx_n', 0.0)
        data['hed_y_delta'] = data.get('hedy', 0.0) - data.get('hedy_n', 0.0)
        data['hed_z_delta'] = data.get('hedz', 0.0) - data.get('hedz_n', 0.0)

        # ── Full causal feature set (117 features) ──
        data = compute_causal_features(data)

        row_dict = {col: data.get(col, 0.0) for col in feature_cols}
        X = pd.DataFrame([row_dict], columns=feature_cols)

        # ── Binary + multiclass probabilities ──
        bin_proba = float(clf_binary.predict_proba(X)[0][1])
        bin_pred = 1 if bin_proba > 0.5 else 0

        mc_proba = clf_multiclass.predict_proba(X)[0]
        mc_pred = int(clf_multiclass.predict(X)[0])

        # ── Per-class threshold decision (matches the actual "final result" rule
        #    evaluated in train_model_enhanced.py / fix_threshold_leak.py) ──
        threshold = optimal_thresholds.get(mc_pred, 0.5)
        pred = 1 if bin_proba > threshold else 0

        # ── Optional hybrid override (off by default — see module docstring) ──
        if USE_HYBRID_OVERRIDE:
            if pred == 0 and mc_pred in HARD_CLASSES:
                pred = 1
            elif pred == 1 and mc_pred == 0 and mc_proba[0] > 0.7:
                pred = 0

        conf = bin_proba if pred == 1 else (1.0 - bin_proba)

        explanation = ""
        if pred == 1:
            print(f"⚠️ Attack flagged on Vehicle {data.get('sender', 'Unknown')} "
                  f"(mc_pred={mc_pred} {ATTACK_NAMES.get(mc_pred,'?')}, "
                  f"bin_proba={bin_proba:.3f}, threshold={threshold:.2f})! Triggering Gemma 3...")
            explanation = explain_anomaly(data, pred, conf)

        return jsonify({
            "vehicle_id": data.get("sender"),
            "prediction": pred,
            "confidence": conf,
            "explanation": explanation,
            "predicted_class": mc_pred,
            "predicted_class_name": ATTACK_NAMES.get(mc_pred, "Unknown"),
            "bin_proba": bin_proba,
            "threshold_used": threshold,
            "posx": data.get("posx"),
            "posy": data.get("posy"),
            "spdx": data.get("spdx"),
            "spdy": data.get("spdy"),
        }), 200

    except Exception as e:
        print(f"Error in /classify: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
