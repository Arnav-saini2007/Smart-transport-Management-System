import pandas as pd
import requests
import json
import time
import os
import warnings
warnings.filterwarnings("ignore")

from driving.kinematic_features import update_kinematics, get_rolling_features, compute_ttc
from driving.crash_rules import evaluate as evaluate_driving

LOG_FILE = "alerts_log.jsonl"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

print("📡 Initializing RSU Edge Node...")

# Check if simulation dataset exists
DATA_PATH = 'data.csv'
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: '{DATA_PATH}' not found! Place data.csv in the working directory.")
    exit()

# 1. Read dataset to simulate live V2X BSM packet stream
df = pd.read_csv(DATA_PATH)

API_URL = "http://127.0.0.1:5001/classify"
LOG_FILE = "alerts_log.jsonl"

print("🚀 RSU Active! Forwarding BSM frames to Model + Gemma 3 API...\n")

# ── Sort chronologically ── the causal features (msg_rate_1s,
# time_since_last_msg) require messages to arrive in real time order,
# exactly as they did during training. The raw CSV row order is NOT
# guaranteed to be chronological.
df = df.sort_values('sendTime').reset_index(drop=True)

# ── If this script is wrapped in an outer loop for a continuous live
# demo (replaying the same file over and over), each pass must advance
# time forward rather than repeating the same sendTime values —
# otherwise every sender's per-message clock appears to freeze or go
# backwards every time the loop restarts, which the model reads as a
# DoS/flooding signature. This offset keeps time monotonically
# increasing across passes. loop_count starts at 0 for a single pass
# and has no effect unless you wrap this script's loop externally.
LOOP_OFFSET = float(df['sendTime'].max() - df['sendTime'].min()) + 1.0
loop_count = int(os.environ.get("RSU_LOOP_COUNT", 0))

for idx, row in df.iterrows():
    # Convert row to dictionary payload for the API
    payload = row.to_dict()
    payload['sendTime'] = payload['sendTime'] + loop_count * LOOP_OFFSET

    # Ensure sender key exists for vehicle identifier
    if 'sender' not in payload:
        payload['sender'] = int(row.get('type', row.get('vehicle_id', idx)))

    # ── Driving / crash-physics layer ──
    # Runs locally (no HTTP round-trip needed — this is cheap enough to run
    # inline, which also matters for phase 3 since the OBU won't always
    # have spare network bandwidth). Uses the _n (noisy/received) fields,
    # since that's what a real receiver actually observes. If your security
    # model instead trains on ground-truth fields, drop the "_n" suffix
    # below and set USE_NOISY_FIELDS = False in kinematic_features.py.
    kin_msg = {
        "sender": payload['sender'],
        "timestamp": payload['sendTime'],
        "posx": payload['posx_n'], "posy": payload['posy_n'],
        "spdx": payload['spdx_n'], "spdy": payload['spdy_n'],
        "aclx": payload['aclx_n'], "acly": payload['acly_n'],
        "hedx": payload['hedx_n'], "hedy": payload['hedy_n'],
    }
    update_kinematics(kin_msg)
    driving_event = evaluate_driving(
        payload['sender'],
        get_rolling_features(payload['sender']),
        compute_ttc(payload['sender']),
    )
    if driving_event:
        print(f"🚨 [DRIVING ALERT] {driving_event.event_type} on Vehicle {payload['sender']} "
              f"| {driving_event.reason}")
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(driving_event.to_alert_dict()) + '\n')

    try:
        # Send telemetry frame to your running model_service.py API
        response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            pred = result.get('prediction', 0)
            conf = result.get('confidence', 0.0)
            veh_id = result.get('vehicle_id', payload['sender'])

            if pred == 1:
                print(f"⚠️  [RSU ALERT] Threat detected on Vehicle {veh_id} | Confidence: {conf:.0%} | Gemma: {result.get('explanation')[:60]}...")
            else:
                print(f"✅ [RSU OK] Frame processed for Vehicle {veh_id}")

            # Append structured result to shared stream log for Streamlit
            result['category'] = result.get('category', 'security')
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(result) + '\n')

        else:
            print(f"❌ API Error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is model_service.py running on port 5001?")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")

    time.sleep(0.5)  # Simulates live stream (2 packets per second)
