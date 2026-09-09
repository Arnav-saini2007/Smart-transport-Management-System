import pandas as pd
df = pd.read_csv(r"D:\VIT\Hackathon\Gdg 2026\VeReMi_Extension Dataset for Misbehaviors in VANETs\mixalldata_clean.csv", nrows=5)
print(df.columns.tolist())
print(df.shape)