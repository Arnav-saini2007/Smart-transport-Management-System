"""
train_model_full_v2.py — Full 3.2M-row training, with per-attack-subtype balancing.

Fixes the issue found in the previous run: undersampling "Attack" as one
proportional pool let high-volume subtypes (13-19) dominate training,
starving rare subtypes (1, 2, 4, 10, 11, 12, 17) of examples. This version
caps EVERY attack subtype (1-19) at the same count before combining, so
the model learns each attack pattern with equal representation.

USAGE: same as before — edit DATA_PATH, then run.
OUTPUT: model_binary_v3_full.pkl, feature_cols_v3_full.pkl, training_summary_v2.txt
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
import time

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

DATA_PATH = r"D:\VIT\Hackathon\Gdg 2026\VeReMi_Extension Dataset for Misbehaviors in VANETs\mixalldata_clean.csv"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

t_start = time.time()
log("="*70)
log("FULL DATASET TRAINING RUN — v2 (per-attack-subtype balanced)")
log("="*70)
log(f"Loading {DATA_PATH} ...")

df = pd.read_csv(DATA_PATH)
log(f"Loaded {len(df):,} rows in {time.time()-t_start:.1f}s")

class_counts = df['class'].value_counts().sort_index()
log(f"\nClass distribution:\n{class_counts.to_string()}")

log("\nEngineering features...")
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
t0 = time.time()
df['msg_rate_1s'] = df.groupby('sender', group_keys=False).apply(trailing_rate)
log(f"  done in {time.time()-t0:.1f}s")

df['spd_mag'] = np.sqrt(df['spdx']**2 + df['spdy']**2)
df['acl_mag'] = np.sqrt(df['aclx']**2 + df['acly']**2)
df['spd_mag_n'] = np.sqrt(df['spdx_n']**2 + df['spdy_n']**2)
df['spd_mag_delta'] = df['spd_mag'] - df['spd_mag_n']

feature_cols = ['posx','posy','posz','posx_n','posy_n','posz_n','spdx','spdy','spdz',
                 'spdx_n','spdy_n','spdz_n','aclx','acly','aclz','aclx_n','acly_n','aclz_n',
                 'hedx','hedy','hedz','hedx_n','hedy_n','hedz_n',
                 'pos_x_delta','pos_y_delta','pos_z_delta',
                 'spd_x_delta','spd_y_delta','spd_z_delta',
                 'acl_x_delta','acl_y_delta','acl_z_delta',
                 'hed_x_delta','hed_y_delta','hed_z_delta',
                 'msg_count_so_far','time_since_last_msg','msg_rate_1s',
                 'spd_mag','acl_mag','spd_mag_n','spd_mag_delta']

df['label'] = (df['class'] != 0).astype(int)

log("\nSplitting by vehicle (sender)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=df['sender']))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
log(f"Senders — train: {train_df['sender'].nunique():,}, test: {test_df['sender'].nunique():,}, "
    f"overlap: {len(set(train_df['sender'])&set(test_df['sender']))}")

# ── NEW: balance WITHIN the attack pool first (cap every subtype 1-19 at the smallest subtype's count) ──
attack_train_pool = train_df[train_df['label']==1]
subtype_counts = attack_train_pool['class'].value_counts()
cap = subtype_counts.min()
log(f"\nAttack subtypes in training pool range from {subtype_counts.min():,} to {subtype_counts.max():,} rows.")
log(f"Capping every attack subtype at {cap:,} rows so no subtype dominates training.")

balanced_attack_parts = []
for cls in sorted(attack_train_pool['class'].unique()):
    part = attack_train_pool[attack_train_pool['class']==cls].sample(n=cap, random_state=RANDOM_STATE)
    balanced_attack_parts.append(part)
attack_train_balanced = pd.concat(balanced_attack_parts)
log(f"Subtype-balanced attack training pool: {len(attack_train_balanced):,} rows "
    f"({len(attack_train_balanced['class'].unique())} subtypes x {cap:,} each)")

# ── Now balance Normal vs Attack at this new total ──
normal_train_pool = train_df[train_df['label']==0]
n_bal = min(len(normal_train_pool), len(attack_train_balanced))
normal_train = normal_train_pool.sample(n=n_bal, random_state=RANDOM_STATE)
attack_train = attack_train_balanced.sample(n=n_bal, random_state=RANDOM_STATE) if len(attack_train_balanced) > n_bal else attack_train_balanced
balanced_train = pd.concat([normal_train, attack_train]).sample(frac=1, random_state=RANDOM_STATE)
log(f"\nFinal balanced training set: {len(balanced_train):,} rows "
    f"({len(normal_train):,} Normal + {len(attack_train):,} Attack)")

log("\nTraining RandomForestClassifier (500 trees, verbose progress)...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=500, max_depth=24, min_samples_leaf=2,
                             max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE,
                             verbose=2)
rf.fit(balanced_train[feature_cols], balanced_train['label'])
train_time = time.time() - t0
log(f"Training took {train_time:.1f}s")

# ── Evaluate on balanced, vehicle-disjoint, natural-proportion test set ──
normal_test_pool = test_df[test_df['label']==0]
attack_test_pool = test_df[test_df['label']==1]
n_bal_test = min(len(normal_test_pool), len(attack_test_pool))
normal_test = normal_test_pool.sample(n=n_bal_test, random_state=RANDOM_STATE)
attack_test = attack_test_pool.sample(n=n_bal_test, random_state=RANDOM_STATE)
bal_test = pd.concat([normal_test, attack_test])

pred = rf.predict(bal_test[feature_cols])
report = classification_report(bal_test['label'], pred, target_names=['Normal','Attack'], digits=4)
acc = accuracy_score(bal_test['label'], pred)
cm = confusion_matrix(bal_test['label'], pred)

log("\n" + "="*70)
log("RESULT — Balanced, vehicle-disjoint test set")
log("="*70)
log(report)
log(f"Accuracy: {acc:.4f}")
log(f"\nConfusion matrix:\n                 Pred Normal   Pred Attack")
log(f"  Actual Normal  {cm[0][0]:10d}   {cm[0][1]:10d}")
log(f"  Actual Attack  {cm[1][0]:10d}   {cm[1][1]:10d}")

test_df_eval = test_df.copy()
test_df_eval['pred'] = rf.predict(test_df_eval[feature_cols])
log("\nPer-class detection rate on held-out vehicles:")
for c in sorted(test_df_eval['class'].unique()):
    sub = test_df_eval[test_df_eval['class']==c]
    target = 0 if c==0 else 1
    correct = (sub['pred']==target).mean()
    log(f"  class {c:2d}  n={len(sub):7d}  correctly-classified={correct*100:5.1f}%")

imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
log("\nTop 10 feature importances:")
log(imp.head(10).to_string())

joblib.dump(rf, 'model_binary_v3_full.pkl')
joblib.dump(feature_cols, 'feature_cols_v3_full.pkl')
log(f"\nSaved model_binary_v3_full.pkl and feature_cols_v3_full.pkl")
log(f"\nTotal script time: {time.time()-t_start:.1f}s")

with open('training_summary_v2.txt', 'w') as f:
    f.write("\n".join(log_lines))

print("\n\n>>> DONE. Send 'training_summary_v2.txt' back. <<<")
