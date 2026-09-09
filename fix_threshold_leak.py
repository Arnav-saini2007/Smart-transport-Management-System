"""
fix_threshold_leak.py — Reuses your already-trained models (no retraining needed),
but fixes the threshold-tuning leak by splitting val/test by VEHICLE, not by row.

Run this in the same folder as your saved .pkl files and the dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import joblib
import warnings, time

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = r"D:\VIT\Hackathon\Gdg 2026\VeReMi_Extension Dataset for Misbehaviors in VANETs\mixalldata_clean.csv"

ATTACK_NAMES = {
    0: 'Normal', 1: 'Const Position', 2: 'Const Pos Offset', 3: 'Random Position', 4: 'Random Pos Offset',
    5: 'Const Speed', 6: 'Const Speed Offset', 7: 'Random Speed', 8: 'Random Speed Offset',
    9: 'Disruptive', 10: 'Data Replay', 11: 'DoS', 12: 'DoS Random',
    13: 'DoS Disruptive', 14: 'Data Replay Sybil', 15: 'Traffic Congestion Sybil',
    16: 'DoS Random Sybil', 17: 'DoS Disruptive Sybil', 18: 'Class18', 19: 'Class19'
}
HARD_CLASSES = {2, 4, 10, 11, 17}

t0 = time.time()
print("Loading data + rebuilding features (same as before — this is the slow part)...")
df = pd.read_csv(DATA_PATH)

# ── Same feature engineering as train_model_enhanced.py ──
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

print(f"Feature engineering done in {time.time()-t0:.1f}s")

# ── Load your ALREADY-TRAINED models (no retraining) ──
lgb_binary = joblib.load('model_binary_enhanced.pkl')
lgb_multiclass = joblib.load('model_multiclass_enhanced.pkl')
feature_cols = joblib.load('feature_cols_enhanced.pkl')

df['label_binary'] = (df['class'] != 0).astype(int)

# ── Reconstruct the SAME train/test vehicle split (same random_state = same split) ──
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=df['sender']))
test_df = df.iloc[test_idx].copy()

# ── FIX: split test vehicles into two DISJOINT groups — one for threshold
#         tuning, one for final evaluation. No vehicle appears in both. ──
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
val_idx, eval_idx = next(gss2.split(test_df, groups=test_df['sender']))
val_df = test_df.iloc[val_idx]
eval_df = test_df.iloc[eval_idx]
overlap = len(set(val_df['sender']) & set(eval_df['sender']))
print(f"Threshold-val vehicles: {val_df['sender'].nunique():,}, "
      f"final-eval vehicles: {eval_df['sender'].nunique():,}, overlap: {overlap}")
assert overlap == 0, "Still leaking!"

# balance both halves for fair evaluation
def balance(d):
    normal = d[d['label_binary']==0]
    attack = d[d['label_binary']==1]
    n = min(len(normal), len(attack))
    return pd.concat([normal.sample(n=n, random_state=RANDOM_STATE),
                       attack.sample(n=n, random_state=RANDOM_STATE)])

val_bal = balance(val_df)
eval_bal = balance(eval_df)
print(f"Threshold-val balanced: {len(val_bal):,}  |  Final-eval balanced: {len(eval_bal):,}")

# ── Tune thresholds on val_bal (vehicle-disjoint from eval_bal now) ──
val_bin_proba = lgb_binary.predict_proba(val_bal[feature_cols])[:, 1]
val_mc_pred = lgb_multiclass.predict(val_bal[feature_cols])

optimal_thresholds = {0: 0.5}
for cls in HARD_CLASSES:
    cls_mask = (val_bal['class'] == cls)
    if cls_mask.sum() > 10:
        cls_proba = val_bin_proba[cls_mask.values]
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (cls_proba > thresh).astype(int)
            if preds.sum() > 0:
                f1 = f1_score(np.ones_like(preds), preds)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, thresh
        optimal_thresholds[cls] = best_thresh
        print(f"  Class {cls} ({ATTACK_NAMES[cls]}): threshold={best_thresh:.2f}, val F1={best_f1:.3f}")

# ── Apply to eval_bal (genuinely unseen vehicles for this threshold step) ──
eval_bin_proba = lgb_binary.predict_proba(eval_bal[feature_cols])[:, 1]
eval_mc_pred = lgb_multiclass.predict(eval_bal[feature_cols])
final_pred = np.zeros(len(eval_bal), dtype=int)
for i, (mc_p, proba) in enumerate(zip(eval_mc_pred, eval_bin_proba)):
    thresh = optimal_thresholds.get(mc_p, 0.5)
    final_pred[i] = 1 if proba > thresh else 0

print("\n" + "="*70)
print("HONEST FINAL RESULT — vehicle-disjoint threshold tuning + eval")
print("="*70)
print(classification_report(eval_bal['label_binary'], final_pred, target_names=['Normal','Attack'], digits=4))
print("Accuracy:", accuracy_score(eval_bal['label_binary'], final_pred))
cm = confusion_matrix(eval_bal['label_binary'], final_pred)
print(f"\nConfusion matrix:\n  Normal->Normal={cm[0][0]}  Normal->Attack={cm[0][1]}")
print(f"  Attack->Normal={cm[1][0]}  Attack->Attack={cm[1][1]}")

print("\nPer-class detection rate (vehicle-disjoint from threshold tuning):")
eval_df_full = eval_df.copy()
eval_df_full['bin_proba'] = lgb_binary.predict_proba(eval_df_full[feature_cols])[:, 1]
eval_df_full['mc_pred'] = lgb_multiclass.predict(eval_df_full[feature_cols])
for c in sorted(eval_df_full['class'].unique()):
    sub = eval_df_full[eval_df_full['class']==c]
    target = 0 if c==0 else 1
    thresh_map = sub['mc_pred'].map(lambda x: optimal_thresholds.get(x, 0.5))
    pred = (sub['bin_proba'] > thresh_map).astype(int)
    correct = (pred == target).mean()
    print(f"  class {c:2d} ({ATTACK_NAMES.get(c,'?'):26s}) n={len(sub):5d}  correctly-classified={correct*100:5.1f}%")
