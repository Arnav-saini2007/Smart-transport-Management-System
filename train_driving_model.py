"""
train_driving_model.py — Fits the "rash driving style" detector mentioned (but
not implemented) in crash_rules.py's original docstring: sustained aggressive
driving behavior over time, as distinct from the instantaneous CRASH/
NEAR_MISS/HARSH_BRAKE/HARSH_CORNER events in crash_rules.py.

WHY THIS IS UNSUPERVISED, NOT A TRAINED CLASSIFIER
====================================================
There is no "this vehicle drove recklessly" label anywhere in this dataset —
VeReMi_Extension's `class` column is a V2X-message misbehavior/attack type
(GPS spoofing, DoS, replay, ...), not a driving-style label. A spoofing
attacker can drive perfectly normally; a legitimate vehicle can brake hard at
an intersection. There is nothing to supervise against.

So this fits an IsolationForest on aggregate per-vehicle driving-style
profiles from genuinely legitimate (class == 0) vehicles only, learning what
"normal" sustained driving behavior looks like, and scores every vehicle by
how anomalous its profile is relative to that population. This is the same
philosophy as the data-driven threshold work already done on crash_rules.py
and the attack-classifier thresholds — let the real distribution decide what
"rare/aggressive" means, rather than guessing a constant.

VALIDATION NOTE: as a sanity check (NOT the detector's intended purpose —
security classification is model_service_v2.py's job), Attack-class vehicles
in this dataset get flagged as "rash" at a 53% rate vs. 2% for Normal-class
vehicles, with a clearly separated mean anomaly score. That's reassuring
evidence the detector responds to genuinely more erratic physical motion
rather than noise — but don't wire this into the security pipeline; it's
scoring driving behavior, not V2X misbehavior, and the correlation here is
a property of this specific simulated dataset, not a general rule.

FEATURES (per vehicle, aggregated over its message history):
    harsh_brake_frac   — fraction of messages that trip HARSH_BRAKE
    harsh_corner_frac  — fraction of messages that trip HARSH_CORNER
    accel_std          — variability of longitudinal acceleration
    accel_min          — hardest single braking event
    jerk_abs_mean       — average abruptness of acceleration changes
    jerk_abs_p95        — near-worst-case abruptness
    yaw_abs_mean         — average turning rate
    yaw_abs_p95          — near-worst-case turning rate
    speed_std            — speed variability (stop-start / erratic pacing)

USAGE:
    Run in the same folder as your dataset (edit DATA_PATH below) and the
    driving/ package. Produces rash_driving_model.pkl, loaded by
    driving/rash_driving.py at runtime.

    python train_driving_model.py
"""
import pandas as pd
import numpy as np
import sys, time, warnings
from sklearn.ensemble import IsolationForest
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
from driving.kinematic_features import update_kinematics, reset_all
from driving.crash_rules import HARSH_BRAKE_DECEL_MPS2, HARSH_CORNER_YAW_RATE_DEG_S

# ── EDIT THIS if your file has a different name/path ──
DATA_PATH = r"D:\VIT\Hackathon\Gdg 2026\VeReMi_Extension Dataset for Misbehaviors in VANETs\mixalldata_clean.csv"

RANDOM_STATE = 42
CONTAMINATION = 0.02   # expected fraction of genuinely rash drivers in a normal population — tune to taste
MIN_MESSAGES_PER_VEHICLE = 10  # need enough history for a meaningful profile

FEATURE_COLS = ['harsh_brake_frac', 'harsh_corner_frac', 'accel_std', 'accel_min',
                 'jerk_abs_mean', 'jerk_abs_p95', 'yaw_abs_mean', 'yaw_abs_p95', 'speed_std']

t0 = time.time()
print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={'class': 'class_'})  # 'class' is a Python keyword — can't use it as an itertuples attribute
df = df.sort_values('sendTime').reset_index(drop=True)
print(f"Loaded {len(df):,} rows, {df['sender'].nunique():,} senders")

# ── Replay every message through the SAME causal kinematics module used in
#    production, building a per-sender running profile as we go. ──
reset_all()
per_sender = {}
for row in df.itertuples(index=False):
    sender = row.sender
    msg = {"sender": sender, "timestamp": row.sendTime,
           "posx": row.posx, "posy": row.posy, "spdx": row.spdx, "spdy": row.spdy,
           "aclx": row.aclx, "acly": row.acly, "hedx": row.hedx, "hedy": row.hedy}
    r = update_kinematics(msg)
    d = per_sender.setdefault(sender, {'accel': [], 'jerk': [], 'yaw': [], 'speed': [], 'class': row.class_})
    d['accel'].append(r.accel)
    d['jerk'].append(r.jerk)
    d['yaw'].append(r.yaw_rate)
    d['speed'].append(r.speed)

print(f"Kinematics replay done in {time.time()-t0:.1f}s")

rows = []
for sender, d in per_sender.items():
    accel = np.array(d['accel']); jerk = np.array(d['jerk'])
    yaw = np.array(d['yaw']); speed = np.array(d['speed'])
    n = len(accel)
    if n < MIN_MESSAGES_PER_VEHICLE:
        continue
    rows.append({
        'sender': sender, 'class': d['class'], 'n_msgs': n,
        'harsh_brake_frac': (accel <= -HARSH_BRAKE_DECEL_MPS2).mean(),
        'harsh_corner_frac': (np.abs(yaw) >= HARSH_CORNER_YAW_RATE_DEG_S).mean(),
        'accel_std': accel.std(),
        'accel_min': accel.min(),
        'jerk_abs_mean': np.abs(jerk).mean(),
        'jerk_abs_p95': np.percentile(np.abs(jerk), 95),
        'yaw_abs_mean': np.abs(yaw).mean(),
        'yaw_abs_p95': np.percentile(np.abs(yaw), 95),
        'speed_std': speed.std(),
    })

profiles = pd.DataFrame(rows)
print(f"Vehicle profiles built: {len(profiles):,} "
      f"(Normal: {(profiles['class']==0).sum():,}, Attack: {(profiles['class']!=0).sum():,})")

normal = profiles[profiles['class'] == 0]
print(f"\nFitting IsolationForest on {len(normal):,} Normal-class vehicle profiles "
      f"(contamination={CONTAMINATION})...")

iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION,
                       random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(normal[FEATURE_COLS])

profiles['anomaly_score'] = -iso.score_samples(profiles[FEATURE_COLS])
profiles['is_rash'] = iso.predict(profiles[FEATURE_COLS]) == -1

print("\nScore distribution by class (sanity check — see docstring):")
print(profiles.groupby(profiles['class'].apply(lambda c: 'Normal' if c == 0 else 'Attack'))
      ['anomaly_score'].describe()[['count', 'mean', 'std', 'min', 'max']])
print("\nRash-flagged rate by class:")
print(profiles.groupby(profiles['class'].apply(lambda c: 'Normal' if c == 0 else 'Attack'))['is_rash'].mean())

# A fixed score threshold (not just the forest's own .predict()) so the
# real-time scorer in driving/rash_driving.py can apply a consistent cutoff
# without needing to call .predict() per-vehicle at serve time.
score_threshold = float(np.percentile(profiles.loc[profiles['class'] == 0, 'anomaly_score'], 100 * (1 - CONTAMINATION)))
print(f"\nScore threshold (Normal-class {100*(1-CONTAMINATION):.0f}th percentile): {score_threshold:.4f}")

joblib.dump({
    'model': iso,
    'feature_cols': FEATURE_COLS,
    'score_threshold': score_threshold,
    'harsh_brake_decel': HARSH_BRAKE_DECEL_MPS2,
    'harsh_corner_yaw': HARSH_CORNER_YAW_RATE_DEG_S,
}, 'rash_driving_model.pkl')

profiles.to_csv('vehicle_driving_profiles.csv', index=False)
print(f"\nSaved rash_driving_model.pkl and vehicle_driving_profiles.csv")
print(f"Total script time: {time.time()-t0:.1f}s")
