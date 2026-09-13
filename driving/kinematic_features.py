"""
kinematic_features.py

Causal (message-by-message, no future leakage) kinematic feature engineering
for crash / near-miss / rash-driving detection.

Built for the VeReMi-style BSM schema used in this project's data.csv:
    type, sendTime, sender, senderPseudo, messageID, class,
    posx, posy, posz, posx_n, posy_n, posz_n,
    spdx, spdy, spdz, spdx_n, spdy_n, spdz_n,
    aclx, acly, aclz, aclx_n, acly_n, aclz_n,
    hedx, hedy, hedz, hedx_n, hedy_n, hedz_n

Non-suffixed fields (posx, spdx, ...) are ground truth from the simulator.
`_n`-suffixed fields are the noisy/received values — what a real RSU or OBU
would actually observe over the air, and what an attacker's misbehavior
would show up in. This module consumes the `_n` fields by default, since
that mirrors real deployment and (most likely) matches what your security
classifier itself was trained on. Flip USE_NOISY_FIELDS to False if your
model_service.py actually trains on ground truth instead.

z-axis (posz/spdz/aclz) is ignored: VeReMi vehicles are simulated on a 2D
road plane (SUMO), so z is ~0 and adds nothing to crash/near-miss physics.

Design mirrors the existing security-layer pattern in model_service_v2.py:
each sender (vehicle) gets its own rolling state, updated one message at a
time, so this runs identically in simulation, on an RSU, or later on an OBU.

Call update_kinematics(msg) once per incoming row, where msg is built like:

    payload = row.to_dict()
    kin_msg = {
        "sender": payload["sender"],
        "timestamp": payload["sendTime"],
        "posx": payload["posx_n"], "posy": payload["posy_n"],
        "spdx": payload["spdx_n"], "spdy": payload["spdy_n"],
        "aclx": payload["aclx_n"], "acly": payload["acly_n"],
        "hedx": payload["hedx_n"], "hedy": payload["hedy_n"],
    }
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZE = 10  # matches the "last 10 messages" convention from train_model_enhanced.py

# Set to False if your security model trains on ground-truth (non-_n) fields
# instead of received/noisy fields — see module docstring.
USE_NOISY_FIELDS = True


@dataclass
class Reading:
    timestamp: float
    posx: float
    posy: float
    spdx: float
    spdy: float
    speed: float          # m/s, magnitude of (spdx, spdy)
    heading_deg: float    # degrees, from heading vector via atan2
    accel: float = 0.0    # m/s^2, longitudinal (accel vector projected onto heading)
    jerk: float = 0.0     # m/s^3
    yaw_rate: float = 0.0  # deg/s


@dataclass
class SenderState:
    """Rolling causal state for one vehicle."""
    history: Deque[Reading] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))

    def last(self) -> Optional[Reading]:
        return self.history[-1] if self.history else None


# Global per-sender state, keyed by sender id. Same pattern as
# model_service_v2.py's causal state dict — module-scope, single dev-server
# process, no locking needed for a single worker.
_sender_states: Dict[str, SenderState] = {}


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference between two headings in degrees, in [-180, 180]."""
    return (a - b + 180) % 360 - 180


def reset_sender(sender) -> None:
    """Clear rolling state for a sender (e.g. on session end / test reset)."""
    _sender_states.pop(sender, None)


def reset_all() -> None:
    _sender_states.clear()


def update_kinematics(msg: dict) -> Reading:
    """
    Feed one incoming message for a sender and get back a Reading populated
    with causal accel / jerk / yaw_rate computed from this and prior messages.

    Expected keys in msg: sender, timestamp, posx, posy, spdx, spdy, aclx,
    acly, hedx, hedy (already resolved to _n or ground-truth by the caller —
    see module docstring for the exact mapping from your data.csv columns).
    """
    sender = msg["sender"]
    state = _sender_states.setdefault(sender, SenderState())
    prev = state.last()

    ts = float(msg["timestamp"])
    posx, posy = float(msg["posx"]), float(msg["posy"])
    spdx, spdy = float(msg["spdx"]), float(msg["spdy"])
    aclx, acly = float(msg["aclx"]), float(msg["acly"])
    hedx, hedy = float(msg["hedx"]), float(msg["hedy"])

    speed = math.hypot(spdx, spdy)

    # Normalize heading vector defensively (VeReMi's hed fields are already
    # unit vectors, but noisy/_n versions may drift slightly off unit length).
    hed_norm = math.hypot(hedx, hedy) or 1.0
    hx, hy = hedx / hed_norm, hedy / hed_norm
    heading_deg = math.degrees(math.atan2(hy, hx))

    # Longitudinal acceleration: project the raw accel vector onto the
    # heading direction. Positive = speeding up, negative = braking.
    # (Lateral component, i.e. accel perpendicular to heading, is implicitly
    # captured instead by yaw_rate below — that's the "cornering hard" signal.)
    accel = aclx * hx + acly * hy

    if prev is None:
        yaw_rate = 0.0
        jerk = 0.0
    else:
        dt = max(ts - prev.timestamp, 1e-3)  # guard div-by-zero on duplicate timestamps
        yaw_rate = _angle_diff_deg(heading_deg, prev.heading_deg) / dt
        jerk = (accel - prev.accel) / dt

    reading = Reading(
        timestamp=ts, posx=posx, posy=posy, spdx=spdx, spdy=spdy,
        speed=speed, heading_deg=heading_deg, accel=accel, jerk=jerk, yaw_rate=yaw_rate,
    )
    state.history.append(reading)
    return reading


def get_rolling_features(sender) -> Optional[dict]:
    """
    Rolling-window features over the last WINDOW_SIZE messages for a sender.
    Same idea as the sequence features in train_model_enhanced.py (rolling
    mean/std/max/min) but for kinematics. Returns None if the sender has no
    history yet.
    """
    state = _sender_states.get(sender)
    if not state or not state.history:
        return None

    accels = [r.accel for r in state.history]
    jerks = [r.jerk for r in state.history]
    yaw_rates = [r.yaw_rate for r in state.history]

    def stats(vals):
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        return mean, var ** 0.5, max(vals), min(vals)

    a_mean, a_std, a_max, a_min = stats(accels)
    j_mean, j_std, j_max, j_min = stats(jerks)
    y_mean, y_std, y_max, y_min = stats(yaw_rates)

    latest = state.history[-1]
    return {
        "sender": sender,
        "timestamp": latest.timestamp,
        "speed": latest.speed,
        "accel": latest.accel,
        "jerk": latest.jerk,
        "yaw_rate": latest.yaw_rate,
        "accel_mean": a_mean, "accel_std": a_std, "accel_max": a_max, "accel_min": a_min,
        "jerk_mean": j_mean, "jerk_std": j_std, "jerk_max": j_max, "jerk_min": j_min,
        "yaw_rate_mean": y_mean, "yaw_rate_std": y_std, "yaw_rate_max": y_max, "yaw_rate_min": y_min,
        "msg_count_so_far": len(state.history),
    }


def compute_ttc(sender, other_senders: Optional[list] = None) -> Optional[dict]:
    """
    Time-to-collision from `sender` to the nearest other currently-tracked
    vehicle, using only the latest known reading per vehicle (causal). Exact
    closed-form geometry (this schema gives us real position + velocity
    vectors, so no bearing-angle approximation is needed here).

    other_senders: optionally restrict the search to this list of sender ids
    (e.g. only vehicles on the same road segment). If omitted, checks every
    other currently tracked sender.
    """
    me = _sender_states.get(sender)
    if not me or not me.history:
        return None
    my_latest = me.history[-1]

    candidates = other_senders if other_senders is not None else list(_sender_states.keys())
    best = None

    for other_id in candidates:
        if other_id == sender:
            continue
        other = _sender_states.get(other_id)
        if not other or not other.history:
            continue
        other_latest = other.history[-1]

        rel_px = other_latest.posx - my_latest.posx
        rel_py = other_latest.posy - my_latest.posy
        rel_vx = other_latest.spdx - my_latest.spdx
        rel_vy = other_latest.spdy - my_latest.spdy

        distance_m = math.hypot(rel_px, rel_py)
        if distance_m < 1e-6:
            continue  # coincident points, undefined geometry

        # Rate of change of distance = (relPos . relVel) / distance.
        # Negative = approaching, so closing_speed (positive when
        # approaching) is the negation of that.
        closing_speed = -(rel_px * rel_vx + rel_py * rel_vy) / distance_m

        if closing_speed <= 0.1:
            continue  # not closing meaningfully, no TTC

        ttc = distance_m / closing_speed

        if best is None or ttc < best["ttc"]:
            best = {
                "other_sender": other_id,
                "distance_m": distance_m,
                "closing_speed_mps": closing_speed,
                "ttc": ttc,
            }

    return best
