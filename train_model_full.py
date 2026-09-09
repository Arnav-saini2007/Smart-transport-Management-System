"""
train_model_full.py — Run this on the full 3.2M-row VeReMi Extension dataset.

USAGE:
    Put this file in the same folder as your full dataset CSV, then either:
    (a) rename your full dataset file to 'full_dataset.csv', or
    (b) edit DATA_PATH below to point at your actual file/folder.

    Then run:
        python train_model_full.py

OUTPUT:
    - model_binary_v2_full.pkl       (trained model)
    - feature_cols_v2_full.pkl       (feature list)
    - training_summary.txt           <-- SEND THIS FILE BACK, it has everything needed
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
import time
import json

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ── EDIT THIS if your file has a different name/path ──
DATA_PATH = "D:\\VIT\\Hackathon\\Gdg 2026\\VeReMi_Extension Dataset for Misbehaviors in VANETs\\mixalldata_clean.csv"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))

t_start = time.time()
log("="*70)
log("FULL DATASET TRAINING RUN")
log("="*70)
log(f"Loading {DATA_PATH} ...")

df = pd.read_csv(DATA_PATH)
log(f"Loaded {len(df):,} rows in {time.time()-t_start:.1f}s")
log(f"Columns: {list(df.columns)}")

class_counts = df['class'].value_counts().sort_index()
log(f"\nClass distribution:\n{class_counts.to_string()}")
normal_n = (df['class']==0).sum()
attack_n = (df['class']!=0).sum()
log(f"\nNormal: {normal_n:,} ({normal_n/len(df)*100:.1f}%)  |  Attack: {attack_n:,} ({attack_n/len(df)*100:.1f}%)")

# ── Feature engineering (identical to the 188K-row pipeline) ──
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

log("Computing trailing message-rate feature (this is the slowest step, be patient on 3.2M rows)...")
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

# ── Vehicle-disjoint split (no sender appears in both train and test) ──
log("\nSplitting by vehicle (sender) — no vehicle appears in both train and test...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=df['sender']))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
log(f"Senders — train: {train_df['sender'].nunique():,}, test: {test_df['sender'].nunique():,}, "
    f"overlap: {len(set(train_df['sender'])&set(test_df['sender']))}")

# ── Balance training set only (undersample majority to match minority) ──
normal_pool = train_df[train_df['label']==0]
attack_pool = train_df[train_df['label']==1]
n_bal = min(len(normal_pool), len(attack_pool))
normal_train = normal_pool.sample(n=n_bal, random_state=RANDOM_STATE)
attack_train = attack_pool.sample(n=n_bal, random_state=RANDOM_STATE)
balanced_train = pd.concat([normal_train, attack_train]).sample(frac=1, random_state=RANDOM_STATE)
log(f"\nBalanced training set: {len(balanced_train):,} rows "
    f"({len(normal_train):,} Normal + {len(attack_train):,} Attack)")

# ── Train ──
log("\nTraining RandomForestClassifier (500 trees)...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=500, max_depth=24, min_samples_leaf=2,
                             max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(balanced_train[feature_cols], balanced_train['label'])
train_time = time.time() - t0
log(f"Training took {train_time:.1f}s")

# ── Evaluate on BALANCED, vehicle-disjoint test set (the honest number) ──
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
log("RESULT — Balanced, vehicle-disjoint test set (compare this to the 96.03% from 188K rows)")
log("="*70)
log(report)
log(f"Accuracy: {acc:.4f}")
log(f"\nConfusion matrix:\n                 Pred Normal   Pred Attack")
log(f"  Actual Normal  {cm[0][0]:10d}   {cm[0][1]:10d}")
log(f"  Actual Attack  {cm[1][0]:10d}   {cm[1][1]:10d}")

# ── Per-class breakdown ──
test_df_eval = test_df.copy()
test_df_eval['pred'] = rf.predict(test_df_eval[feature_cols])
log("\nPer-class detection rate on held-out vehicles:")
for c in sorted(test_df_eval['class'].unique()):
    sub = test_df_eval[test_df_eval['class']==c]
    target = 0 if c==0 else 1
    correct = (sub['pred']==target).mean()
    log(f"  class {c:2d}  n={len(sub):7d}  correctly-classified={correct*100:5.1f}%")

# ── Feature importances ──
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
log("\nTop 10 feature importances:")
log(imp.head(10).to_string())

# ── Save everything ──
joblib.dump(rf, 'model_binary_v2_full.pkl')
joblib.dump(feature_cols, 'feature_cols_v2_full.pkl')
log(f"\nSaved model_binary_v2_full.pkl and feature_cols_v2_full.pkl")
log(f"\nTotal script time: {time.time()-t_start:.1f}s")

with open('training_summary.txt', 'w') as f:
    f.write("\n".join(log_lines))

print("\n\n>>> DONE. Send the file 'training_summary.txt' back — it has everything needed. <<<")
