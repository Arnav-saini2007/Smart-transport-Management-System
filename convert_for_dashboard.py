# convert_for_dashboard.py
import json
import pandas as pd

records = []
with open('alerts_log.jsonl') as f:
    for line in f:
        a = json.loads(line)
        records.append({
            "vehicle_id": f"V{a['vehicle_id']}",
            "speed": (a.get("spdx") or 0)**2 + (a.get("spdy") or 0)**2,
            "status": "Attack" if a["prediction"] == 1 else "Normal",
            "confidence": a["confidence"] * 100,
            "explanation": a.get("explanation", ""),
            "attack_type": "GPS Spoofing" if a["prediction"] == 1 else "",
            "risk_level": "High" if a["prediction"] == 1 else "Low",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

df = pd.DataFrame(records)
df.to_csv('data/predictions.csv', index=False)
print(f"Converted {len(df)} rows")
print(df['status'].value_counts())