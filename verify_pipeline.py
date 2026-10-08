"""
verify_pipeline.py — Full verification of the enhanced binary+multiclass pipeline
with the CORRECTED per-class thresholds (optimal_thresholds_v2.pkl).

Run this in the same folder as:
    - mixalldata_clean.csv (or edit DATA_PATH below)
    - model_binary_enhanced.pkl
    - model_multiclass_enhanced.pkl
    - feature_cols_enhanced.pkl
    - optimal_thresholds_v2.pkl       (from fix_threshold_leak_v2.py)
    - optimal_thresholds.pkl          (optional — the OLD buggy one, for comparison)

It does NOT retrain anything. It:
  1. Rebuilds the same 117 features used everywhere else in this project.
  2. Reconstructs the SAME vehicle-disjoint train/val/eval split used by
     fix_threshold_leak_v2.py (same random_state=42), so this evaluates on
     the `eval_df` half — vehicles that were NEVER used for threshold tuning.
  3. Applies the exact decision rule used in model_service_v2.py:
         threshold = optimal_thresholds.get(multiclass_prediction, 0.5)
         prediction = 1 if binary_proba > threshold else 0
  4. Reports, for BOTH the old and new threshold files (if both are present):
       - Overall accuracy / precision / recall / F1 (balanced + natural-proportion)
       - Confusion matrix
       - Per-class detection rate (all 20 classes)
       - Per-class REAL precision/recall/F1 — i.e. computed against actual
         Normal rows as negatives, not the broken all-positive-slice metric
         the original threshold search used
       - The multiclass classifier's own standalone accuracy/macro-F1
  5. Writes everything to verification_summary.txt so you don't have to
     scroll back through console output.

USAGE:
    python verify_pipeline.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                              f1_score, precision_score, recall_score)
import joblib
import warnings, time

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ── EDIT THIS if your file has a different name/path ──
DATA_PATH = r"D:\VIT\Hackathon\Gdg 2026\VeReMi_Extension Dataset for Misbehaviors in VANETs\mixalldata_clean.csv"

HARD_CLASSES = {2, 4, 10, 11, 17}
ATTACK_NAMES = {
    0: 'Normal', 1: 'Const Position', 2: 'Const Pos Offset', 3: 'Random Position', 4: 'Random Pos Offset',
    5: 'Const Speed', 6: 'Const Speed Offset', 7: 'Random Speed', 8: 'Random Speed Offset',
    9: 'Disruptive', 10: 'Data Replay', 11: 'DoS', 12: 'DoS Random',
    13: 'DoS Disruptive', 14: 'Data Replay Sybil', 15: 'Traffic Congestion Sybil',
    16: 'DoS Random Sybil', 17: 'DoS Disruptive Sybil', 18: 'Class18', 19: 'Class19'
}

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

t_start = time.time()
log("="*80)
log("PIPELINE VERIFICATION — enhanced model + corrected per-class thresholds")
log("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD + REBUILD FEATURES (identical to train_model_enhanced.py / fix_threshold_leak_v2.py)
# ─────────────────────────────────────────────────────────────────────────────
log(f"\nLoading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
log(f"Loaded {len(df):,} rows")

for a,b in [('posx','posx_n'),('posy','posy_n'),('posz','posz_n'),
            ('spdx','spdx_n'),('spdy','spdy_n'),('spdz','spdz_n'),
            ('aclx','aclx_n'),('acly','acly_n'),('aclz','aclz_n'),
            ('hedx','hedx_n'),('hedy','hedy_n'),('hedz','hedz_n')]:
    df[f'{a[:3]}_{a[3]}_delta'] = df[a] - df[b]

df = df.sort_values(['sender','sendTime']).reset_index(drop=True)
df['msg_count_so_far'] = df.groupby('sender').cumcount() + 1
df['time_since_last_msg'] = df.groupby('sender')['sendTime'].diff().fillna(999)

def trailing_rate(g):
    t = g['sendTime'].values
    counts = np.zeros(len(t)); j = 0
    for i in range(len(t)):
        while t[i] - t[j] > 1.0:
            j += 1
        counts[i] = i - j + 1
    return pd.Series(counts, index=g.index)

log("Computing trailing message-rate feature...")
df['msg_rate_1s'] = df.groupby('sender', group_keys=False).apply(trailing_rate)

df['spd_mag'] = np.sqrt(df['spdx']**2 + df['spdy']**2)
df['acl_mag'] = np.sqrt(df['aclx']**2 + df['acly']**2)
df['spd_mag_n'] = np.sqrt(df['spdx_n']**2 + df['spdy_n']**2)
df['spd_mag_delta'] = df['spd_mag'] - df['spd_mag_n']

df['pos_delta_mag'] = np.sqrt(df['pos_x_delta']**2 + df['pos_y_delta']**2 + df['pos_z_delta']**2)
df['spd_delta_mag'] = np.sqrt(df['spd_x_delta']**2 + df['spd_y_delta']**2 + df['spd_z_delta']**2)
df['acl_delta_mag'] = np.sqrt(df['acl_x_delta']**2 + df['acl_y_delta']**2 + df['acl_z_delta']**2)
df['hed_delta_mag'] = np.sqrt(df['hed_x_delta']**2 + df['hed_y_delta']**2 + df['hed_z_delta']**2)

def delta_direction_change(group, prefix):
    dx = group[f'{prefix}_x_delta'].values
    dy = group[f'{prefix}_y_delta'].values
    dz = group[f'{prefix}_z_delta'].values
    mag = np.sqrt(dx**2 + dy**2 + dz**2)
    cos_sim = np.ones(len(group))
    for i in range(1, len(group)):
        if mag[i] > 1e-6 and mag[i-1] > 1e-6:
            cos_sim[i] = (dx[i]*dx[i-1] + dy[i]*dy[i-1] + dz[i]*dz[i-1]) / (mag[i] * mag[i-1])
    return pd.Series(1 - cos_sim, index=group.index)

log("Computing direction-change features...")
for prefix in ['pos', 'spd', 'acl', 'hed']:
    df[f'{prefix}_dir_change'] = df.groupby('sender', group_keys=False).apply(lambda g: delta_direction_change(g, prefix))

WINDOW = 5
for col in ['pos_x_delta', 'pos_y_delta', 'pos_z_delta',
            'spd_x_delta', 'spd_y_delta', 'spd_z_delta',
            'acl_x_delta', 'acl_y_delta', 'acl_z_delta']:
    df[f'{col}_roll_var'] = df.groupby('sender')[col].transform(lambda x: x.rolling(WINDOW, min_periods=1).var())
    df[f'{col}_roll_mean'] = df.groupby('sender')[col].transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())

df['pos_delta_roll_var_mean'] = df[['pos_x_delta_roll_var', 'pos_y_delta_roll_var', 'pos_z_delta_roll_var']].mean(axis=1)
df['spd_delta_roll_var_mean'] = df[['spd_x_delta_roll_var', 'spd_y_delta_roll_var', 'spd_z_delta_roll_var']].mean(axis=1)
df['acl_delta_roll_var_mean'] = df[['acl_x_delta_roll_var', 'acl_y_delta_roll_var', 'acl_z_delta_roll_var']].mean(axis=1)

SEQ_WINDOW = 10
seq_cols = ['pos_delta_mag', 'spd_delta_mag', 'acl_delta_mag', 'hed_delta_mag',
            'pos_dir_change', 'spd_dir_change', 'acl_dir_change', 'hed_dir_change',
            'time_since_last_msg', 'msg_rate_1s',
            'spd_mag_delta', 'pos_x_delta', 'pos_y_delta', 'pos_z_delta']

log("Computing rolling sequence features (slow step)...")
for col in seq_cols:
    if col in df.columns:
        df[f'{col}_roll_mean_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(lambda x: x.rolling(SEQ_WINDOW, min_periods=1).mean())
        df[f'{col}_roll_std_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(lambda x: x.rolling(SEQ_WINDOW, min_periods=1).std())
        df[f'{col}_roll_max_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(lambda x: x.rolling(SEQ_WINDOW, min_periods=1).max())
        df[f'{col}_roll_min_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(lambda x: x.rolling(SEQ_WINDOW, min_periods=1).min())

for col in ['pos_delta_mag', 'spd_delta_mag', 'time_since_last_msg', 'msg_rate_1s']:
    if col in df.columns:
        df[f'{col}_trend_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=2).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=True))

for col in ['pos_delta_mag', 'spd_delta_mag', 'time_since_last_msg']:
    if col in df.columns:
        df[f'{col}_autocorr_1'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=3).apply(
                lambda y: np.corrcoef(y[:-1], y[1:])[0,1] if len(y) > 2 and np.std(y) > 1e-6 else 0, raw=True))

feature_cols = joblib.load('feature_cols_enhanced.pkl')
df[feature_cols] = df[feature_cols].fillna(0)
df['label_binary'] = (df['class'] != 0).astype(int)
log(f"Feature engineering done in {time.time()-t_start:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODELS + THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
clf_binary = joblib.load('model_binary_enhanced.pkl')
clf_multiclass = joblib.load('model_multiclass_enhanced.pkl')

thresholds_new = None
thresholds_old = None
try:
    thresholds_new = joblib.load('optimal_thresholds_v2.pkl')
    log(f"\nLoaded CORRECTED thresholds (optimal_thresholds_v2.pkl): {thresholds_new}")
except FileNotFoundError:
    log("\n⚠️  optimal_thresholds_v2.pkl not found. Run fix_threshold_leak_v2.py first. Aborting.")
    raise SystemExit(1)

try:
    thresholds_old = joblib.load('optimal_thresholds.pkl')
    log(f"Loaded OLD thresholds for comparison (optimal_thresholds.pkl): {thresholds_old}")
except FileNotFoundError:
    log("(optimal_thresholds.pkl not found — skipping old-vs-new comparison)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RECONSTRUCT THE SAME VEHICLE-DISJOINT SPLIT USED BY fix_threshold_leak_v2.py
# ─────────────────────────────────────────────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=df['sender']))
test_df = df.iloc[test_idx].copy()

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
val_idx, eval_idx = next(gss2.split(test_df, groups=test_df['sender']))
val_df = test_df.iloc[val_idx].copy()
eval_df = test_df.iloc[eval_idx].copy()
overlap = len(set(val_df['sender']) & set(eval_df['sender']))
log(f"\nVal vehicles: {val_df['sender'].nunique():,}  Eval vehicles: {eval_df['sender'].nunique():,}  overlap: {overlap}")
assert overlap == 0, "Vehicle overlap detected — split is leaking!"
log("This script evaluates ONLY on eval_df — vehicles never touched by threshold tuning.")

# Score once
eval_df['bin_proba'] = clf_binary.predict_proba(eval_df[feature_cols])[:, 1]
eval_df['mc_pred'] = clf_multiclass.predict(eval_df[feature_cols])
mc_proba_full = clf_multiclass.predict_proba(eval_df[feature_cols])


def apply_thresholds(d, thresholds):
    """Mirrors model_service_v2.py's decision rule exactly (hybrid override OFF)."""
    thresh_map = d['mc_pred'].map(lambda x: thresholds.get(x, 0.5))
    return (d['bin_proba'] > thresh_map).astype(int)


def balance(d):
    normal = d[d['label_binary']==0]
    attack = d[d['label_binary']==1]
    n = min(len(normal), len(attack))
    return pd.concat([normal.sample(n=n, random_state=RANDOM_STATE),
                       attack.sample(n=n, random_state=RANDOM_STATE)])


def full_report(d, thresholds, label):
    log("\n" + "="*80)
    log(f"REPORT — {label}")
    log("="*80)

    pred_full = apply_thresholds(d, thresholds)
    log(f"\n[Natural class proportions, n={len(d):,}]")
    log(classification_report(d['label_binary'], pred_full, target_names=['Normal','Attack'], digits=4))
    log(f"Accuracy: {accuracy_score(d['label_binary'], pred_full):.4f}")
    cm = confusion_matrix(d['label_binary'], pred_full)
    log(f"Confusion matrix:\n                 Pred Normal   Pred Attack")
    log(f"  Actual Normal  {cm[0][0]:10d}   {cm[0][1]:10d}")
    log(f"  Actual Attack  {cm[1][0]:10d}   {cm[1][1]:10d}")

    d_bal = balance(d)
    pred_bal = apply_thresholds(d_bal, thresholds)
    log(f"\n[Balanced 50/50 Normal:Attack, n={len(d_bal):,}]")
    log(classification_report(d_bal['label_binary'], pred_bal, target_names=['Normal','Attack'], digits=4))
    log(f"Accuracy: {accuracy_score(d_bal['label_binary'], pred_bal):.4f}")

    log("\nPer-class detection rate (all 20 classes):")
    for c in sorted(d['class'].unique()):
        sub = d[d['class']==c]
        target = 0 if c==0 else 1
        thresh_map = sub['mc_pred'].map(lambda x: thresholds.get(x, 0.5))
        pred = (sub['bin_proba'] > thresh_map).astype(int)
        correct = (pred == target).mean()
        log(f"  class {c:2d} ({ATTACK_NAMES.get(c,'?'):26s}) n={len(sub):6d}  correctly-classified={correct*100:5.1f}%")

    log("\nPer-class REAL precision/recall/F1 (hard classes only, matched against actual Normal rows):")
    normal_pool = d[d['class'] == 0]
    for c in sorted(HARD_CLASSES):
        sub_attack = d[d['class'] == c]
        thresh = thresholds.get(c, 0.5)
        n = min(len(sub_attack), len(normal_pool))
        if n == 0:
            continue
        pos = sub_attack.sample(n=n, random_state=RANDOM_STATE) if len(sub_attack) > n else sub_attack
        neg = normal_pool.sample(n=n, random_state=RANDOM_STATE)
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        proba = pd.concat([pos, neg])['bin_proba'].values
        preds = (proba > thresh).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        log(f"  class {c:2d} ({ATTACK_NAMES[c]:26s}) threshold={thresh:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")

    return pred_full


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN REPORTS
# ─────────────────────────────────────────────────────────────────────────────
full_report(eval_df, thresholds_new, "CORRECTED thresholds (optimal_thresholds_v2.pkl)")

if thresholds_old is not None:
    full_report(eval_df, thresholds_old, "OLD thresholds (optimal_thresholds.pkl) — for comparison")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MULTICLASS HEAD'S OWN STANDALONE ACCURACY (independent of binary threshold)
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "="*80)
log("MULTICLASS CLASSIFIER — standalone performance (20-way, not threshold-dependent)")
log("="*80)
log(classification_report(
    eval_df['class'], eval_df['mc_pred'],
    labels=sorted(ATTACK_NAMES.keys()),
    target_names=[ATTACK_NAMES[c] for c in sorted(ATTACK_NAMES.keys())],
    digits=4, zero_division=0
))

log(f"\nTotal script time: {time.time()-t_start:.1f}s")
with open('verification_summary.txt', 'w') as f:
    f.write("\n".join(log_lines))

print("\n\n>>> DONE. Full results written to 'verification_summary.txt'. <<<")
