"""
rash_driving.py

Causal (message-by-message, no future leakage) "rash driving style" scorer —
sustained aggressive behavior over time, as distinct from the instantaneous
CRASH/NEAR_MISS/HARSH_BRAKE/HARSH_CORNER events in crash_rules.py.

Loads rash_driving_model.pkl (built offline by train_driving_model.py — an
IsolationForest fit on genuinely legitimate vehicles' aggregate driving-style
profiles) and, per sender, maintains a rolling window of recent kinematic
readings to compute the SAME aggregate features the model was trained on,
scoring the vehicle's current driving style against it.

This deliberately uses a LONGER window than kinematic_features.py's
WINDOW_SIZE=10 (which is tuned for instant-event detection) — rash driving is
a pattern over many messages, not a single reading, so it needs more history
to say anything meaningful. ROLLING_WINDOW below controls this.

Usage (same pattern as crash_rules.py):
    from driving.kinematic_features import update_kinematics
    from driving.rash_driving import score_driving_style

    reading = update_kinematics(msg)   # call this first, as usual
    event = score_driving_style(msg["sender"], reading)
    if event:
        alerts_log.write(event.to_alert_dict())
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 50          # messages of history per vehicle before/while scoring
MIN_MESSAGES_TO_SCORE = 15   # don't score a vehicle until it has a meaningful history
RESCORE_EVERY = 10           # only recompute the (relatively expensive) forest score every N messages
ALERT_COOLDOWN_MESSAGES = 30  # don't re-alert on the same vehicle more than once per this many messages

_MODEL_PATH = os.environ.get("RASH_DRIVING_MODEL_PATH", "rash_driving_model.pkl")
_artifact = None  # lazy-loaded so this module can be imported before the model file exists


def _load_artifact():
    global _artifact
    if _artifact is None:
        _artifact = joblib.load(_MODEL_PATH)
    return _artifact


@dataclass
class SenderStyleState:
    accel: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    jerk: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    yaw: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    speed: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    msg_count: int = 0
    last_alert_at: int = -10**9  # msg_count value at last alert, for cooldown
    last_score: float = 0.0


_states: Dict[str, SenderStyleState] = {}


def reset_sender(sender) -> None:
    _states.pop(sender, None)


def reset_all() -> None:
    _states.clear()


@dataclass
class DrivingStyleEvent:
    sender: str
    timestamp: float
    event_type: str = "RASH_DRIVING_STYLE"
    severity: int = 1  # informational/pattern-level, not an instantaneous emergency
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    def to_alert_dict(self) -> dict:
        return {
            "sender": self.sender,
            "timestamp": self.timestamp,
            "category": "driving",
            "event_type": self.event_type,
            "severity": self.severity,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def _compute_features(state: SenderStyleState, harsh_brake_decel: float, harsh_corner_yaw: float) -> dict:
    accel = np.array(state.accel); jerk = np.array(state.jerk)
    yaw = np.array(state.yaw); speed = np.array(state.speed)
    return {
        'harsh_brake_frac': float((accel <= -harsh_brake_decel).mean()),
        'harsh_corner_frac': float((np.abs(yaw) >= harsh_corner_yaw).mean()),
        'accel_std': float(accel.std()),
        'accel_min': float(accel.min()),
        'jerk_abs_mean': float(np.abs(jerk).mean()),
        'jerk_abs_p95': float(np.percentile(np.abs(jerk), 95)),
        'yaw_abs_mean': float(np.abs(yaw).mean()),
        'yaw_abs_p95': float(np.percentile(np.abs(yaw), 95)),
        'speed_std': float(speed.std()),
    }


def score_driving_style(sender, reading) -> Optional[DrivingStyleEvent]:
    """
    Call once per message, AFTER kinematic_features.update_kinematics() has
    produced `reading` for this message. Returns a DrivingStyleEvent only when
    a vehicle's sustained driving-style anomaly score crosses the trained
    threshold AND the per-vehicle alert cooldown has elapsed — otherwise None.
    """
    artifact = _load_artifact()
    model = artifact['model']
    feature_cols = artifact['feature_cols']
    threshold = artifact['score_threshold']
    harsh_brake_decel = artifact['harsh_brake_decel']
    harsh_corner_yaw = artifact['harsh_corner_yaw']

    state = _states.setdefault(sender, SenderStyleState())
    state.accel.append(reading.accel)
    state.jerk.append(reading.jerk)
    state.yaw.append(reading.yaw_rate)
    state.speed.append(reading.speed)
    state.msg_count += 1

    if state.msg_count < MIN_MESSAGES_TO_SCORE:
        return None
    if state.msg_count % RESCORE_EVERY != 0:
        return None

    feats = _compute_features(state, harsh_brake_decel, harsh_corner_yaw)
    X = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)
    score = float(-model.score_samples(X)[0])
    state.last_score = score

    if score < threshold:
        return None
    if state.msg_count - state.last_alert_at < ALERT_COOLDOWN_MESSAGES:
        return None  # already alerted on this vehicle recently — avoid spam

    state.last_alert_at = state.msg_count
    return DrivingStyleEvent(
        sender=sender,
        timestamp=reading.timestamp,
        reason=(
            f"Sustained aggressive driving pattern over last {len(state.accel)} messages: "
            f"harsh-brake rate {feats['harsh_brake_frac']*100:.0f}%, "
            f"harsh-corner rate {feats['harsh_corner_frac']*100:.0f}%, "
            f"jerk p95 {feats['jerk_abs_p95']:.1f} m/s^3 "
            f"(anomaly score {score:.3f} vs. threshold {threshold:.3f})."
        ),
        evidence={**feats, 'anomaly_score': score, 'threshold': threshold},
    )
