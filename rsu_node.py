import pandas as pd
import requests
import json
import time
import os
import warnings
warnings.filterwarnings("ignore")

from driving.kinematic_features import update_kinematics, get_rolling_features, compute_ttc
from driving.crash_rules import evaluate as evaluate_driving
from driving.rash_driving import score_driving_style

# rash_driving_model.pkl is produced by train_driving_model.py — if it hasn't
# been run yet, don't crash the whole RSU node over it; just skip rash-driving
# scoring and keep doing instant-event crash detection + security classification.
try:
    score_driving_style  # (lazy-loads the model on first real call, not here)
    import joblib as _joblib
    _joblib.load('rash_driving_model.pkl')  # fail fast with a clear message if missing
    RASH_DRIVING_ENABLED = True
except FileNotFoundError:
    RASH_DRIVING_ENABLED = False
    print("⚠️  rash_driving_model.pkl not found — rash-driving-style scoring disabled. "
          "Run train_driving_model.py to enable it.")

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
    # Runs locally (no HTTP round-trip needed). Uses the GROUND-TRUTH
    # (non-"_n") position/speed/accel/heading fields.
    #
    # CHANGED: this used to read the "_n"-suffixed fields on the assumption
    # that they were noisy/received readings ("what a real receiver
    # observes"). They are not — in this dataset they're a much
    # smaller-magnitude noise VECTOR, not a usable position/speed/accel
    # signal (confirmed against the original VeReMi paper's description of
    # a separate "position noise vector" field, and empirically: posx_n
    # ranges ~0-5 while posx ranges ~0-1500, correlation ~0.05). Feeding
    # "_n" fields into update_kinematics() produced near-zero speed/accel
    # for every vehicle and made every crash_rules threshold unreachable.
    # Ground-truth fields give real, physically meaningful kinematics.
    # If you later get access to a genuine noisy-position column (e.g. a
    # real GPS-error-augmented field, not this dataset's noise vector),
    # point this back at that column instead.
    kin_msg = {
        "sender": payload['sender'],
        "timestamp": payload['sendTime'],
        "posx": payload['posx'], "posy": payload['posy'],
        "spdx": payload['spdx'], "spdy": payload['spdy'],
        "aclx": payload['aclx'], "acly": payload['acly'],
        "hedx": payload['hedx'], "hedy": payload['hedy'],
    }
    reading = update_kinematics(kin_msg)
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

    # ── Rash-driving-style layer (sustained pattern, not an instant event) ──
    # Independent of the instant-event check above — a vehicle can trip both,
    # neither, or just one, so this is logged separately rather than folded
    # into evaluate_driving()'s single-highest-severity result.
    if RASH_DRIVING_ENABLED:
        style_event = score_driving_style(payload['sender'], reading)
        if style_event:
            print(f"🧭 [DRIVING STYLE] {style_event.event_type} on Vehicle {payload['sender']} "
                  f"| {style_event.reason}")
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(style_event.to_alert_dict()) + '\n')

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
