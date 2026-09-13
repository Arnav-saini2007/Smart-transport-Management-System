"""
crash_rules.py

Rule-based (physics-threshold) detector for CRASH, NEAR_MISS, and HARSH_BRAKE
events. Deliberately not ML: these are instantaneous physical events with
well-understood signatures, so explainable thresholds beat a learned model
here — same reasoning your security layer used for going causal-only, just
applied to "when is a threshold breach actually a rule vs. a pattern".

Rash-driving *style* (sustained aggressive behavior over time) is a separate,
harder problem better suited to the ML layer in train_driving_model.py —
this file only handles the sharp, physically unambiguous events.

Usage:
    from driving.kinematic_features import update_kinematics, get_rolling_features, compute_ttc
    from driving.crash_rules import evaluate

    reading = update_kinematics(msg)
    features = get_rolling_features(msg["sender"])
    ttc_info = compute_ttc(msg["sender"])
    event = evaluate(msg["sender"], features, ttc_info)
    if event:
        alerts_log.write(event)   # same alerts_log.jsonl your security layer uses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Thresholds — start here, then tune against your simulated data distribution
# the same way fix_threshold_leak.py tunes per-class thresholds: look at the
# actual accel/jerk distributions you generate, pick thresholds that sit
# clearly above normal-driving noise, then validate on held-out sim runs.
# All accel values in m/s^2. 1g ~= 9.81 m/s^2.
# ---------------------------------------------------------------------------

G = 9.81

HARSH_BRAKE_DECEL_MPS2 = 0.4 * G     # ~0.4g — noticeable hard braking
CRASH_DECEL_MPS2 = 0.6 * G           # ~0.6g — sudden, severe deceleration spike
CRASH_POST_SPEED_MPS = 0.6           # ~2 km/h — near-zero speed right after the spike
CRASH_JERK_MPS3 = 15.0               # very abrupt onset (jerk = rate of change of accel)

NEAR_MISS_TTC_S = 2.0                # time-to-collision under 2s = near-miss
NEAR_MISS_DISTANCE_CAP_M = 60.0      # ignore TTC computed from very far away (noisy bearing math)

HARSH_CORNER_YAW_RATE_DEG_S = 25.0   # sharp swerve / harsh turn, feeds rash-driving features too

SEVERITY_ORDER = {"HARSH_BRAKE": 1, "HARSH_CORNER": 1, "NEAR_MISS": 2, "CRASH": 3}


@dataclass
class DrivingEvent:
    sender: str
    timestamp: float
    event_type: str          # "CRASH" | "NEAR_MISS" | "HARSH_BRAKE" | "HARSH_CORNER"
    severity: int            # 1 = warning, 2 = urgent, 3 = critical
    reason: str              # human-readable, feeds directly into the Gemma prompt layer
    evidence: dict           # raw numbers behind the call, for the dashboard/alert card

    def to_alert_dict(self) -> dict:
        """Matches the alerts_log.jsonl schema style — add a `category` field
        so the security layer and driving layer can share one log/dashboard."""
        return {
            "sender": self.sender,
            "timestamp": self.timestamp,
            "category": "driving",
            "event_type": self.event_type,
            "severity": self.severity,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def _check_crash(sender: str, features: dict) -> Optional[DrivingEvent]:
    accel = features["accel"]
    speed = features["speed"]
    jerk = features["jerk"]

    if accel <= -CRASH_DECEL_MPS2 and speed <= CRASH_POST_SPEED_MPS:
        return DrivingEvent(
            sender=sender, timestamp=features["timestamp"], event_type="CRASH", severity=3,
            reason=(
                f"Sudden deceleration of {abs(accel):.1f} m/s^2 "
                f"({abs(accel)/G:.2f}g) followed by near-zero speed "
                f"({speed:.1f} m/s) — consistent with a collision."
            ),
            evidence={"accel": accel, "speed_after": speed, "jerk": jerk},
        )
    return None


def _check_harsh_brake(sender: str, features: dict) -> Optional[DrivingEvent]:
    accel = features["accel"]
    if accel <= -HARSH_BRAKE_DECEL_MPS2:
        return DrivingEvent(
            sender=sender, timestamp=features["timestamp"], event_type="HARSH_BRAKE", severity=1,
            reason=f"Harsh braking detected: {abs(accel):.1f} m/s^2 ({abs(accel)/G:.2f}g).",
            evidence={"accel": accel},
        )
    return None


def _check_harsh_corner(sender: str, features: dict) -> Optional[DrivingEvent]:
    yaw_rate = features["yaw_rate"]
    if abs(yaw_rate) >= HARSH_CORNER_YAW_RATE_DEG_S:
        return DrivingEvent(
            sender=sender, timestamp=features["timestamp"], event_type="HARSH_CORNER", severity=1,
            reason=f"Sharp swerve/turn detected: yaw rate {yaw_rate:.1f} deg/s.",
            evidence={"yaw_rate": yaw_rate},
        )
    return None


def _check_near_miss(sender: str, ttc_info: Optional[dict], timestamp: float) -> Optional[DrivingEvent]:
    if not ttc_info:
        return None
    if ttc_info["distance_m"] > NEAR_MISS_DISTANCE_CAP_M:
        return None
    if ttc_info["ttc"] <= NEAR_MISS_TTC_S:
        return DrivingEvent(
            sender=sender, timestamp=timestamp, event_type="NEAR_MISS", severity=2,
            reason=(
                f"Time-to-collision with vehicle {ttc_info['other_sender']} is "
                f"{ttc_info['ttc']:.1f}s at {ttc_info['distance_m']:.0f}m distance, "
                f"closing at {ttc_info['closing_speed_mps']:.1f} m/s."
            ),
            evidence=ttc_info,
        )
    return None


def evaluate(sender: str, features: Optional[dict], ttc_info: Optional[dict] = None) -> Optional[DrivingEvent]:
    """
    Run all rules for one sender's latest reading and return the single
    highest-severity event, if any (CRASH beats NEAR_MISS beats HARSH_*,
    since a message that looks like both a crash and a harsh brake should
    only raise one alert).

    `features` should come from kinematic_features.get_rolling_features(sender).
    `ttc_info` should come from kinematic_features.compute_ttc(sender).
    """
    if features is None:
        return None

    candidates = [
        _check_crash(sender, features),
        _check_near_miss(sender, ttc_info, features["timestamp"]),
        _check_harsh_brake(sender, features),
        _check_harsh_corner(sender, features),
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None

    return max(candidates, key=lambda e: SEVERITY_ORDER[e.event_type])
