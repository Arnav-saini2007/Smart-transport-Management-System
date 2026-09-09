"""
train_model_enhanced.py — Enhanced training with sequence features, multi-class,
enriched deltas, LightGBM, and per-subtype threshold tuning.

Fixes for the 5 failing subtypes (4, 10, 11, 17, 2):
- Sequence features catch replay (10) and subtle drift (4, 2)
- Multi-class head learns subtype-specific patterns
- Enriched deltas add magnitude/direction/neighbor-consistency signals
- LightGBM captures non-linear interactions better than RF
- Per-subtype thresholds optimize recall for hard classes
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import lightgbm as lgb
import joblib
import warnings
import time
import json

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

DATA_PATH = r"D:\VIT\Hackathon\Gdg 2026\VeReMi_Extension Dataset for Misbehaviors in VANETs\mixalldata_clean.csv"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

# ─── Attack type names ───
ATTACK_NAMES = {
    0: 'Normal',
    1: 'Const Position', 2: 'Const Pos Offset', 3: 'Random Position', 4: 'Random Pos Offset',
    5: 'Const Speed', 6: 'Const Speed Offset', 7: 'Random Speed', 8: 'Random Speed Offset',
    9: 'Disruptive', 10: 'Data Replay', 11: 'DoS', 12: 'DoS Random',
    13: 'DoS Disruptive', 14: 'Data Replay Sybil', 15: 'Traffic Congestion Sybil',
    16: 'DoS Random Sybil', 17: 'DoS Disruptive Sybil', 18: 'Class18', 19: 'Class19'
}

# ─── Hard classes that need extra attention ───
HARD_CLASSES = {2, 4, 10, 11, 17}

t_start = time.time()
log("="*70)
log("ENHANCED TRAINING — Sequence features + Multi-class + LightGBM")
log("="*70)
log(f"Loading {DATA_PATH} ...")

df = pd.read_csv(DATA_PATH)
log(f"Loaded {len(df):,} rows in {time.time()-t_start:.1f}s")
log(f"Columns: {list(df.columns)}")

class_counts = df['class'].value_counts().sort_index()
log(f"\nClass distribution:\n{class_counts.to_string()}")

# ─── Feature Engineering ───
log("\nEngineering features...")

# Original deltas
for a,b in [('posx','posx_n'),('posy','posy_n'),('posz','posz_n'),
            ('spdx','spdx_n'),('spdy','spdy_n'),('spdz','spdz_n'),
            ('aclx','aclx_n'),('acly','acly_n'),('aclz','aclz_n'),
            ('hedx','hedx_n'),('hedy','hedy_n'),('hedz','hedz_n')]:
    df[f'{a[:3]}_{a[3]}_delta'] = df[a] - df[b]

df = df.sort_values(['sender','sendTime']).reset_index(drop=True)

# Causal temporal features
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
t0 = time.time()
df['msg_rate_1s'] = df.groupby('sender', group_keys=False).apply(trailing_rate)
log(f"  done in {time.time()-t0:.1f}s")

# Speed/accel magnitudes
df['spd_mag'] = np.sqrt(df['spdx']**2 + df['spdy']**2)
df['acl_mag'] = np.sqrt(df['aclx']**2 + df['acly']**2)
df['spd_mag_n'] = np.sqrt(df['spdx_n']**2 + df['spdy_n']**2)
df['spd_mag_delta'] = df['spd_mag'] - df['spd_mag_n']

# ─── NEW: Enriched delta features ───
log("Adding enriched delta features...")

# Delta magnitudes (Euclidean)
df['pos_delta_mag'] = np.sqrt(df['pos_x_delta']**2 + df['pos_y_delta']**2 + df['pos_z_delta']**2)
df['spd_delta_mag'] = np.sqrt(df['spd_x_delta']**2 + df['spd_y_delta']**2 + df['spd_z_delta']**2)
df['acl_delta_mag'] = np.sqrt(df['acl_x_delta']**2 + df['acl_y_delta']**2 + df['acl_z_delta']**2)
df['hed_delta_mag'] = np.sqrt(df['hed_x_delta']**2 + df['hed_y_delta']**2 + df['hed_z_delta']**2)

# Delta direction changes (cosine similarity between consecutive delta vectors)
def delta_direction_change(group, prefix):
    dx = group[f'{prefix}_x_delta'].values
    dy = group[f'{prefix}_y_delta'].values
    dz = group[f'{prefix}_z_delta'].values
    mag = np.sqrt(dx**2 + dy**2 + dz**2)
    cos_sim = np.ones(len(group))
    for i in range(1, len(group)):
        if mag[i] > 1e-6 and mag[i-1] > 1e-6:
            cos_sim[i] = (dx[i]*dx[i-1] + dy[i]*dy[i-1] + dz[i]*dz[i-1]) / (mag[i] * mag[i-1])
    return pd.Series(1 - cos_sim, index=group.index)  # 1 - cos = angle change

for prefix in ['pos', 'spd', 'acl', 'hed']:
    df[f'{prefix}_dir_change'] = df.groupby('sender', group_keys=False).apply(lambda g: delta_direction_change(g, prefix))

# Neighbor consistency: how many neighbors agree (simplified - using delta variance across messages)
# For each sender, compute rolling variance of deltas over last 5 messages
WINDOW = 5
for col in ['pos_x_delta', 'pos_y_delta', 'pos_z_delta',
            'spd_x_delta', 'spd_y_delta', 'spd_z_delta',
            'acl_x_delta', 'acl_y_delta', 'acl_z_delta']:
    df[f'{col}_roll_var'] = df.groupby('sender')[col].transform(
        lambda x: x.rolling(WINDOW, min_periods=1).var()
    )
    df[f'{col}_roll_mean'] = df.groupby('sender')[col].transform(
        lambda x: x.rolling(WINDOW, min_periods=1).mean()
    )

# Aggregate rolling stats
df['pos_delta_roll_var_mean'] = df[['pos_x_delta_roll_var', 'pos_y_delta_roll_var', 'pos_z_delta_roll_var']].mean(axis=1)
df['spd_delta_roll_var_mean'] = df[['spd_x_delta_roll_var', 'spd_y_delta_roll_var', 'spd_z_delta_roll_var']].mean(axis=1)
df['acl_delta_roll_var_mean'] = df[['acl_x_delta_roll_var', 'acl_y_delta_roll_var', 'acl_z_delta_roll_var']].mean(axis=1)

# ─── NEW: Sequence features (rolling over last N messages) ───
log("Computing sequence features (rolling statistics)...")
SEQ_WINDOW = 10
seq_cols = ['pos_delta_mag', 'spd_delta_mag', 'acl_delta_mag', 'hed_delta_mag',
            'pos_dir_change', 'spd_dir_change', 'acl_dir_change', 'hed_dir_change',
            'time_since_last_msg', 'msg_rate_1s',
            'spd_mag_delta', 'pos_x_delta', 'pos_y_delta', 'pos_z_delta']

for col in seq_cols:
    if col in df.columns:
        df[f'{col}_roll_mean_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=1).mean()
        )
        df[f'{col}_roll_std_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=1).std()
        )
        df[f'{col}_roll_max_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=1).max()
        )
        df[f'{col}_roll_min_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=1).min()
        )

# Trend features (linear slope over window)
for col in ['pos_delta_mag', 'spd_delta_mag', 'time_since_last_msg', 'msg_rate_1s']:
    if col in df.columns:
        df[f'{col}_trend_{SEQ_WINDOW}'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=2).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=True
            )
        )

# Autocorrelation at lag 1 (catches replay)
for col in ['pos_delta_mag', 'spd_delta_mag', 'time_since_last_msg']:
    if col in df.columns:
        df[f'{col}_autocorr_1'] = df.groupby('sender')[col].transform(
            lambda x: x.rolling(SEQ_WINDOW, min_periods=3).apply(
                lambda y: np.corrcoef(y[:-1], y[1:])[0,1] if len(y) > 2 and np.std(y) > 1e-6 else 0, raw=True
            )
        )

log(f"  Total columns after sequence features: {len(df.columns)}")

# ─── Final feature column list ───
base_features = ['posx','posy','posz','posx_n','posy_n','posz_n','spdx','spdy','spdz',
                 'spdx_n','spdy_n','spdz_n','aclx','acly','aclz','aclx_n','acly_n','aclz_n',
                 'hedx','hedy','hedz','hedx_n','hedy_n','hedz_n',
                 'pos_x_delta','pos_y_delta','pos_z_delta',
                 'spd_x_delta','spd_y_delta','spd_z_delta',
                 'acl_x_delta','acl_y_delta','acl_z_delta',
                 'hed_x_delta','hed_y_delta','hed_z_delta',
                 'msg_count_so_far','time_since_last_msg','msg_rate_1s',
                 'spd_mag','acl_mag','spd_mag_n','spd_mag_delta']

enriched_features = ['pos_delta_mag', 'spd_delta_mag', 'acl_delta_mag', 'hed_delta_mag',
                     'pos_dir_change', 'spd_dir_change', 'acl_dir_change', 'hed_dir_change',
                     'pos_delta_roll_var_mean', 'spd_delta_roll_var_mean', 'acl_delta_roll_var_mean']

# Sequence features
seq_features = [c for c in df.columns if any(s in c for s in ['_roll_mean_', '_roll_std_', '_roll_max_', '_roll_min_', '_trend_', '_autocorr_'])]

feature_cols = base_features + enriched_features + seq_features
# Remove any NaN columns
feature_cols = [c for c in feature_cols if c in df.columns and df[c].notna().any()]

log(f"\nTotal features: {len(feature_cols)}")
log(f"  Base: {len(base_features)}")
log(f"  Enriched: {len(enriched_features)}")
log(f"  Sequence: {len(seq_features)}")

# Fill NaNs with 0 (for rolling features at start of sequences)
df[feature_cols] = df[feature_cols].fillna(0)

# Labels
df['label_binary'] = (df['class'] != 0).astype(int)
df['label_multiclass'] = df['class'].astype(int)

# ─── Vehicle-disjoint split ───
log("\nSplitting by vehicle (sender)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=df['sender']))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
log(f"Senders — train: {train_df['sender'].nunique():,}, test: {test_df['sender'].nunique():,}, overlap: {len(set(train_df['sender'])&set(test_df['sender']))}")

# ─── Per-subtype balanced training (for multi-class) ───
log("\nBalancing training data per subtype...")
attack_train = train_df[train_df['label_binary']==1]
normal_train = train_df[train_df['label_binary']==0]

# Cap each attack subtype at the minimum count
subtype_counts = attack_train['class'].value_counts()
cap = subtype_counts.min()
log(f"Attack subtypes range: {subtype_counts.min():,} to {subtype_counts.max():,}, capping at {cap:,}")

balanced_attack_parts = []
for cls in sorted(attack_train['class'].unique()):
    part = attack_train[attack_train['class']==cls].sample(n=cap, random_state=RANDOM_STATE)
    balanced_attack_parts.append(part)
attack_train_balanced = pd.concat(balanced_attack_parts)

# Balance normal to match total attack
n_bal = min(len(normal_train), len(attack_train_balanced))
normal_train_bal = normal_train.sample(n=n_bal, random_state=RANDOM_STATE)
attack_train_bal = attack_train_balanced.sample(n=n_bal, random_state=RANDOM_STATE)

balanced_train = pd.concat([normal_train_bal, attack_train_bal]).sample(frac=1, random_state=RANDOM_STATE)
log(f"Balanced training: {len(balanced_train):,} rows ({len(normal_train_bal):,} Normal + {len(attack_train_bal):,} Attack)")

# ─── Train Binary Classifier (LightGBM) ───
log("\nTraining Binary LightGBM...")
t0 = time.time()

lgb_binary = lgb.LGBMClassifier(
    n_estimators=800,
    max_depth=16,
    learning_rate=0.03,
    num_leaves=255,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
    class_weight='balanced'
)

lgb_binary.fit(balanced_train[feature_cols], balanced_train['label_binary'])
log(f"Binary training took {time.time()-t0:.1f}s")

# ─── Train Multi-class Classifier (LightGBM) ───
log("\nTraining Multi-class LightGBM...")
t0 = time.time()

# For multi-class, use all training data (not just balanced) but with class weights
train_multiclass = train_df.copy()
class_weights = {c: len(train_multiclass) / (len(train_multiclass[train_multiclass['class']==c]) * 20) for c in train_multiclass['class'].unique()}
class_weights[0] = 1.0  # Normal weight = 1
sample_weights = train_multiclass['class'].map(class_weights).values

lgb_multiclass = lgb.LGBMClassifier(
    n_estimators=600,
    max_depth=14,
    learning_rate=0.04,
    num_leaves=127,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
    objective='multiclass',
    num_class=20
)

lgb_multiclass.fit(train_multiclass[feature_cols], train_multiclass['label_multiclass'], sample_weight=sample_weights)
log(f"Multi-class training took {time.time()-t0:.1f}s")

# ─── Evaluate on Balanced Test Set ───
log("\nEvaluating on balanced, vehicle-disjoint test set...")
normal_test_pool = test_df[test_df['label_binary']==0]
attack_test_pool = test_df[test_df['label_binary']==1]
n_bal_test = min(len(normal_test_pool), len(attack_test_pool))
log(f"  Normal test pool: {len(normal_test_pool):,}, Attack test pool: {len(attack_test_pool):,}, balanced n: {n_bal_test:,}")
normal_test = normal_test_pool.sample(n=n_bal_test, random_state=RANDOM_STATE)
attack_test = attack_test_pool.sample(n=n_bal_test, random_state=RANDOM_STATE)
bal_test = pd.concat([normal_test, attack_test]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
log(f"  Balanced test set: {len(bal_test):,} rows (Normal: {(bal_test['label_binary']==0).sum():,}, Attack: {(bal_test['label_binary']==1).sum():,})")

# Binary predictions
bin_pred = lgb_binary.predict(bal_test[feature_cols])
bin_proba = lgb_binary.predict_proba(bal_test[feature_cols])[:, 1]

# Multi-class predictions
mc_pred = lgb_multiclass.predict(bal_test[feature_cols])
mc_proba = lgb_multiclass.predict_proba(bal_test[feature_cols])

# ─── Hybrid: Use multi-class to refine binary for hard classes ───
log("\nApplying hybrid prediction (binary + multi-class refinement)...")
hybrid_pred = bin_pred.copy()
hard_class_mask = np.isin(mc_pred, list(HARD_CLASSES))
# If multi-class predicts a hard attack class but binary says normal, trust multi-class
hybrid_pred[(bin_pred == 0) & hard_class_mask] = 1

# Also: if binary says attack but multi-class says normal with high confidence, flip
mc_normal_conf = mc_proba[:, 0]
hybrid_pred[(bin_pred == 1) & (mc_pred == 0) & (mc_normal_conf > 0.7)] = 0

# ─── Per-subtype threshold tuning ───
log("\nPer-subtype threshold optimization...")
# For each hard class, find optimal threshold on validation split (use shuffled balanced test)
val_split = int(0.5 * len(bal_test))
val_df = bal_test.iloc[:val_split]
test_eval_df = bal_test.iloc[val_split:]
log(f"  Validation set: {len(val_df):,} (Normal: {(val_df['label_binary']==0).sum():,}, Attack: {(val_df['label_binary']==1).sum():,})")
log(f"  Test eval set: {len(test_eval_df):,} (Normal: {(test_eval_df['label_binary']==0).sum():,}, Attack: {(test_eval_df['label_binary']==1).sum():,})")

val_bin_proba = lgb_binary.predict_proba(val_df[feature_cols])[:, 1]
val_mc_proba = lgb_multiclass.predict_proba(val_df[feature_cols])
val_mc_pred = lgb_multiclass.predict(val_df[feature_cols])

# Find optimal threshold per hard class
optimal_thresholds = {0: 0.5}  # default
for cls in HARD_CLASSES:
    cls_mask = (val_df['class'] == cls)
    if cls_mask.sum() > 10:
        cls_proba = val_bin_proba[cls_mask]
        # Find threshold that gives best F1 for this class
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (cls_proba > thresh).astype(int)
            if preds.sum() > 0:
                f1 = f1_score(np.ones_like(preds), preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        optimal_thresholds[cls] = best_thresh
        log(f"  Class {cls} ({ATTACK_NAMES[cls]}): optimal threshold = {best_thresh:.2f}, F1 = {best_f1:.3f}")

# Apply per-class thresholds on test set
test_bin_proba = lgb_binary.predict_proba(test_eval_df[feature_cols])[:, 1]
test_mc_pred = lgb_multiclass.predict(test_eval_df[feature_cols])
final_pred = np.zeros(len(test_eval_df), dtype=int)

for i in range(len(test_eval_df)):
    mc_p = test_mc_pred[i]
    if mc_p in optimal_thresholds:
        final_pred[i] = 1 if test_bin_proba[i] > optimal_thresholds[mc_p] else 0
    else:
        final_pred[i] = 1 if test_bin_proba[i] > 0.5 else 0

# ─── Final Evaluation ───
log("\n" + "="*70)
log("FINAL RESULTS — Hybrid + Per-Class Thresholds")
log("="*70)

report = classification_report(test_eval_df['label_binary'], final_pred, target_names=['Normal','Attack'], digits=4)
acc = accuracy_score(test_eval_df['label_binary'], final_pred)
cm = confusion_matrix(test_eval_df['label_binary'], final_pred)

log(report)
log(f"Accuracy: {acc:.4f}")
log(f"\nConfusion matrix:")
log(f"                 Pred Normal   Pred Attack")
log(f"  Actual Normal  {cm[0][0]:10d}   {cm[0][1]:10d}")
log(f"  Actual Attack  {cm[1][0]:10d}   {cm[1][1]:10d}")

# Per-class breakdown
log("\nPer-class detection rate on held-out vehicles:")
test_df_eval = test_df.copy()
test_df_eval['bin_proba'] = lgb_binary.predict_proba(test_df_eval[feature_cols])[:, 1]
test_df_eval['mc_pred'] = lgb_multiclass.predict(test_df_eval[feature_cols])

for c in sorted(test_df_eval['class'].unique()):
    sub = test_df_eval[test_df_eval['class']==c]
    target = 0 if c==0 else 1
    
    # Apply per-class threshold
    if c in optimal_thresholds:
        pred = (sub['bin_proba'] > optimal_thresholds[c]).astype(int)
    else:
        pred = (sub['bin_proba'] > 0.5).astype(int)
    
    correct = (pred == target).mean()
    log(f"  class {c:2d} ({ATTACK_NAMES.get(c,'?'):26s}) n={len(sub):5d}  correctly-classified={correct*100:5.1f}%")

# Feature importances
imp_binary = pd.Series(lgb_binary.feature_importances_, index=feature_cols).sort_values(ascending=False)
log("\nTop 15 Binary feature importances:")
log(imp_binary.head(15).to_string())

imp_multiclass = pd.Series(lgb_multiclass.feature_importances_, index=feature_cols).sort_values(ascending=False)
log("\nTop 15 Multi-class feature importances:")
log(imp_multiclass.head(15).to_string())

# ─── Save everything ───
joblib.dump(lgb_binary, 'model_binary_enhanced.pkl')
joblib.dump(lgb_multiclass, 'model_multiclass_enhanced.pkl')
joblib.dump(feature_cols, 'feature_cols_enhanced.pkl')
joblib.dump(optimal_thresholds, 'optimal_thresholds.pkl')
log(f"\nSaved models and artifacts")

log(f"\nTotal script time: {time.time()-t_start:.1f}s")

with open('training_summary_enhanced.txt', 'w') as f:
    f.write("\n".join(log_lines))

print("\n\n>>> DONE. Check 'training_summary_enhanced.txt' for full results. <<<")