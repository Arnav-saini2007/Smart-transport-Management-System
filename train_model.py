import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
df = pd.read_csv(r'D:\VIT\Hackathon\Gdg 2026\VeReMi_Extension Dataset for Misbehaviors in VANETs\train_sample.csv')

for a,b in [('posx','posx_n'),('posy','posy_n'),('posz','posz_n'),
            ('spdx','spdx_n'),('spdy','spdy_n'),('spdz','spdz_n'),
            ('aclx','aclx_n'),('acly','acly_n'),('aclz','aclz_n'),
            ('hedx','hedx_n'),('hedy','hedy_n'),('hedz','hedz_n')]:
    df[f'{a[:3]}_{a[3]}_delta'] = df[a] - df[b]

df = df.sort_values(['sender','sendTime']).reset_index(drop=True)

# CAUSAL versions only — computable live, message-by-message, with a running dict keyed by sender
df['msg_count_so_far'] = df.groupby('sender').cumcount() + 1
df['time_since_last_msg'] = df.groupby('sender')['sendTime'].diff().fillna(999)
# rolling count of messages from this sender in the trailing 1-second window (causal, deployable)
def trailing_rate(g):
    t = g['sendTime'].values
    counts = np.zeros(len(t))
    j = 0
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

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(df, groups=df['sender']))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

normal_train = train_df[train_df['label']==0]
attack_train = train_df[train_df['label']==1].sample(n=len(normal_train), random_state=RANDOM_STATE)
balanced_train = pd.concat([normal_train, attack_train]).sample(frac=1, random_state=RANDOM_STATE)

rf = RandomForestClassifier(n_estimators=500, max_depth=24, min_samples_leaf=2,
                             max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(balanced_train[feature_cols], balanced_train['label'])

normal_test = test_df[test_df['label']==0]
attack_test = test_df[test_df['label']==1].sample(n=len(normal_test), random_state=RANDOM_STATE)
bal_test = pd.concat([normal_test, attack_test])

pred = rf.predict(bal_test[feature_cols])
print("=== RandomForest — CAUSAL features only, vehicle-disjoint balanced test set ===")
print(classification_report(bal_test['label'], pred, target_names=['Normal','Attack'], digits=4))
print("Accuracy:", accuracy_score(bal_test['label'], pred))

imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 8 feature importances:")
print(imp.head(8))

joblib.dump(rf, 'model_binary_v2.pkl')
joblib.dump(feature_cols, 'feature_cols_v2.pkl')
print("\nSaved v5 (causal, deployment-safe) model")

# Per-class breakdown with final model
test_df_full = test_df.copy()
test_df_full['pred'] = rf.predict(test_df_full[feature_cols])
names = {0:'Normal',1:'Const Position',2:'Const Pos Offset',3:'Random Position',4:'Random Pos Offset',
         5:'Const Speed',6:'Const Speed Offset',7:'Random Speed',8:'Random Speed Offset',
         9:'Disruptive',10:'Data Replay',11:'DoS',12:'DoS Random',13:'DoS Disruptive',
         14:'Data Replay Sybil',15:'Traffic Congestion Sybil',16:'DoS Random Sybil',17:'DoS Disruptive Sybil',
         18:'Class18',19:'Class19'}
print("\nPer-class detection rate on held-out vehicles (final causal model):")
for c in sorted(test_df_full['class'].unique()):
    sub = test_df_full[test_df_full['class']==c]
    target = 0 if c==0 else 1
    correct = (sub['pred']==target).mean()
    print(f"  class {c:2d} ({names.get(c,'?'):26s}) n={len(sub):5d}  correctly-classified={correct*100:5.1f}%")
