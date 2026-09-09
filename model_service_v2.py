from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from gemma_layer import explain_anomaly

app = Flask(__name__)

print("1. Loading model artifacts...")
clf = joblib.load('model_binary_v2.pkl')
feature_cols = joblib.load('feature_cols_v2.pkl')
print(f"✅ Loaded {len(feature_cols)} required features.")

# ── Per-sender running state (in-memory, resets if the service restarts) ──
# Needed for msg_count_so_far / time_since_last_msg / msg_rate_1s, which must
# be computed causally (only from messages seen so far) to match how the
# model was trained and to stay realistic for a live RSU deployment.
sender_msg_count = defaultdict(int)          # total messages seen so far, per sender
sender_last_time = {}                        # last sendTime seen, per sender
sender_recent_times = defaultdict(deque)     # trailing 1s window of timestamps, per sender


def compute_causal_features(data):
    sender = data.get('sender')
    send_time = float(data.get('sendTime', 0.0))

    # ── Loop/replay guard ──
    # If this sender's timestamp did not advance (equal to or earlier than
    # the last one we saw for it), we're seeing a replay/loop restart
    # rather than a genuinely new message arriving later in time. Reset
    # this sender's running state so stale history doesn't get misread as
    # a message flood. This works regardless of how the client
    # (rsu_node.py) is invoked or looped, since it only depends on what
    # this server has itself already observed for this sender — no
    # coordination with the client is required.
    last_t = sender_last_time.get(sender)
    if last_t is not None and send_time <= last_t:
        sender_msg_count[sender] = 0
        sender_recent_times[sender].clear()
        last_t = None  # treat this message as this sender's first again

    # msg_count_so_far
    sender_msg_count[sender] += 1
    data['msg_count_so_far'] = sender_msg_count[sender]

    # time_since_last_msg
    data['time_since_last_msg'] = (send_time - last_t) if last_t is not None else 999.0
    sender_last_time[sender] = send_time

    # msg_rate_1s — trailing 1-second message count for this sender
    dq = sender_recent_times[sender]
    dq.append(send_time)
    while dq and send_time - dq[0] > 1.0:
        dq.popleft()
    data['msg_rate_1s'] = len(dq)

    return data


@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json or {}

        # Recompute the same delta features used in training
        data['pos_x_delta'] = data.get('posx', 0.0) - data.get('posx_n', 0.0)
        data['pos_y_delta'] = data.get('posy', 0.0) - data.get('posy_n', 0.0)
        data['pos_z_delta'] = data.get('posz', 0.0) - data.get('posz_n', 0.0)
        data['spd_x_delta'] = data.get('spdx', 0.0) - data.get('spdx_n', 0.0)
        data['spd_y_delta'] = data.get('spdy', 0.0) - data.get('spdy_n', 0.0)
        data['spd_z_delta'] = data.get('spdz', 0.0) - data.get('spdz_n', 0.0)
        data['acl_x_delta'] = data.get('aclx', 0.0) - data.get('aclx_n', 0.0)
        data['acl_y_delta'] = data.get('acly', 0.0) - data.get('acly_n', 0.0)
        data['acl_z_delta'] = data.get('aclz', 0.0) - data.get('aclz_n', 0.0)
        data['hed_x_delta'] = data.get('hedx', 0.0) - data.get('hedx_n', 0.0)
        data['hed_y_delta'] = data.get('hedy', 0.0) - data.get('hedy_n', 0.0)
        data['hed_z_delta'] = data.get('hedz', 0.0) - data.get('hedz_n', 0.0)

        # NEW: speed/accel magnitude features
        spdx, spdy = data.get('spdx', 0.0), data.get('spdy', 0.0)
        spdx_n, spdy_n = data.get('spdx_n', 0.0), data.get('spdy_n', 0.0)
        aclx, acly = data.get('aclx', 0.0), data.get('acly', 0.0)
        data['spd_mag'] = float(np.sqrt(spdx**2 + spdy**2))
        data['acl_mag'] = float(np.sqrt(aclx**2 + acly**2))
        data['spd_mag_n'] = float(np.sqrt(spdx_n**2 + spdy_n**2))
        data['spd_mag_delta'] = data['spd_mag'] - data['spd_mag_n']

        # NEW: causal per-sender temporal features (message-rate based — targets DoS-type attacks)
        data = compute_causal_features(data)

        row_dict = {col: data.get(col, 0.0) for col in feature_cols}
        X = pd.DataFrame([row_dict], columns=feature_cols)

        pred = int(clf.predict(X)[0])
        conf = float(clf.predict_proba(X)[0][pred])

        explanation = ""
        if pred == 1:
            print(f"⚠️ Attack flagged on Vehicle {data.get('sender', 'Unknown')}! Triggering Gemma 3...")
            explanation = explain_anomaly(data, pred, conf)

        return jsonify({
            "vehicle_id": data.get("sender"),
            "prediction": pred,
            "confidence": conf,
            "explanation": explanation,
            "posx": data.get("posx"),
            "posy": data.get("posy"),
            "spdx": data.get("spdx"),
            "spdy": data.get("spdy"),
        }), 200

    except Exception as e:
        print(f"Error in /classify: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
