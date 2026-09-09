import ollama
 
 
def explain_anomaly(row, prediction, confidence):
    """
    row: a dict (or pandas Series) for one BSM record, including the features
         the v2 model actually uses — position/speed/accel/heading, their deltas,
         magnitude features, and the causal per-sender temporal features
         (msg_count_so_far, time_since_last_msg, msg_rate_1s) that catch
         DoS/frequency-based attacks the original prompt never mentioned.
    prediction: 0 (normal) or 1 (attack)
    confidence: model's predicted probability for that class
    """
    if prediction == 0:
        status = "classified as NORMAL"
    else:
        status = "FLAGGED AS ANOMALOUS"
 
    sender_id = row.get('sender', 'Unknown Vehicle')
 
    # ── Pull the temporal/frequency signals the v2 model actually looks at ──
    msg_rate_1s = row.get('msg_rate_1s', None)
    time_since_last = row.get('time_since_last_msg', None)
    msg_count_so_far = row.get('msg_count_so_far', None)
    spd_mag = row.get('spd_mag', None)
    spd_mag_delta = row.get('spd_mag_delta', None)
 
    # 999.0 is a SENTINEL meaning "no prior message is known from this
    # sender yet" — it is NOT a real elapsed duration. It's the default
    # for a vehicle's first-ever observed message (e.g. it will be 999
    # for every row in a single-message-per-vehicle demo file). Treating
    # it as a literal "long silence" produces a plausible-sounding but
    # factually wrong explanation, since every brand-new sender gets this
    # same placeholder regardless of whether anything is actually wrong.
    is_first_message = (time_since_last is not None and time_since_last >= 999.0)
 
    # Build a plain-language hint about WHICH signal likely drove the flag,
    # so Gemma doesn't default to talking about position/heading when the
    # real trigger was message frequency (DoS-type behavior) — and doesn't
    # invent a "long gap" narrative out of the first-message sentinel.
    signal_hints = []
    if msg_rate_1s is not None and msg_rate_1s > 5:
        signal_hints.append(
            f"This vehicle sent {msg_rate_1s:.0f} messages in the last second alone — "
            f"far more than normal V2X update rates, which usually points to a flooding/DoS-style attack "
            f"rather than a falsified position or speed."
        )
    if time_since_last is not None and not is_first_message and 0 <= time_since_last < 0.05:
        signal_hints.append(
            f"Only {time_since_last*1000:.0f} milliseconds passed since this vehicle's previous message, "
            f"which is unusually rapid for legitimate BSM broadcasting."
        )
    if spd_mag_delta is not None and abs(spd_mag_delta) > 15:
        signal_hints.append(
            f"The reported speed differs from the sensor-estimated speed by {abs(spd_mag_delta):.1f} units, "
            f"suggesting the broadcast speed may not match the vehicle's actual motion."
        )
 
    if signal_hints:
        signal_context = "Likely trigger signals:\n- " + "\n- ".join(signal_hints)
    elif is_first_message:
        signal_context = (
            "Note: this is the first message seen from this sender, so no message-timing history is "
            "available yet — do NOT describe this as a long gap, silence, or delay. If this message is "
            "flagged, the reason is a content-level mismatch in this single message's position, speed, "
            "acceleration, or heading values, not anything about timing."
        )
    else:
        signal_context = (
            "No single dominant frequency/magnitude signal stood out — the flag is likely based on a "
            "combination of subtler position/speed/heading deviations."
        )
 
    prompt = f"""You are a V2X network security assistant monitoring vehicle safety messages (BSMs) from an RSU.
A message from vehicle {sender_id} was {status} with {confidence:.0%} model confidence.
 
Message details:
- Position: ({row.get('posx', 0):.1f}, {row.get('posy', 0):.1f})
- Speed vector: ({row.get('spdx', 0):.2f}, {row.get('spdy', 0):.2f})  |  Speed magnitude: {spd_mag if spd_mag is not None else 'n/a'}
- Acceleration: ({row.get('aclx', 0):.2f}, {row.get('acly', 0):.2f})
- Heading: ({row.get('hedx', 0):.2f}, {row.get('hedy', 0):.2f})
- Messages from this sender in the last second: {msg_rate_1s if msg_rate_1s is not None else 'n/a'}
- Time since this sender's previous message: {'first message ever seen from this sender — no timing history yet' if is_first_message else (f'{time_since_last:.3f}s' if time_since_last is not None else 'n/a')}
- Total messages seen from this sender so far: {msg_count_so_far if msg_count_so_far is not None else 'n/a'}
 
{signal_context}
 
In 1-2 short sentences, explain to a traffic operator why this message looks {"suspicious" if prediction == 1 else "normal"}.
If the likely trigger is message frequency/timing (a flooding or DoS-style pattern), say so explicitly rather than
describing it as a position or speed anomaly. Never describe a first-ever message from a sender as a "long gap",
"delay", or "silence" — that phrasing is only valid when a sender has a genuine prior message to compare against.
Be concise and non-technical."""
 
    response = ollama.chat(model='gemma3:4b', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']
 