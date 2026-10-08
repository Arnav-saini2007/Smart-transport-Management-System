"""
crash_rules.py

Rule-based (physics-threshold) detector for CRASH, NEAR_MISS, HARSH_BRAKE, and
HARSH_CORNER events.

v2 — DATA-VALIDATED THRESHOLDS
================================
The thresholds below were re-derived by replaying ~529K real BSM messages
(4,000 vehicle-disjoint senders) from mixalldata_clean.csv through this exact
module and comparing against the constants that used to be here. Findings:

  - IMPORTANT: this requires GROUND-TRUTH kinematics (posx/posy/spdx/spdy/
    aclx/acly/hedx/hedy — the non-"_n" columns). The "_n"-suffixed columns in
    this dataset are NOT noisy/received position-speed-accel readings as the
    original module docstring assumed — they're a separate, much smaller-
    magnitude noise VECTOR (confirmed against the original VeReMi paper,
    which describes "position vector", "speed vector" AND a separate
    "position noise vector"/"speed noise vector" as distinct reception-log
    fields; r(posx, posx_n) ≈ 0.05 in this data, nowhere near what a noisy
    version of the same signal would show). Feeding "_n" fields into this
    module produces near-zero accel/speed for every vehicle and makes every
    rule below unreachable. USE_NOISY_FIELDS is kept here for compatibility
    but should stay False until/unless you have an actual noisy-position
    column to point it at.

  - CRASH_DECEL_MPS2 (0.6g): SUMO's car-following model physically caps
    deceleration at ≈4.5 m/s² in this dataset, so this threshold is, and
    should remain, UNREACHABLE here — there are no real collisions in a
    SUMO traffic simulation to calibrate against. This is not a bug; it
    means this rule is currently untestable rather than miscalibrated, and
    it's kept at the literature-reasonable 0.6g for when real crash-level
    physics (actual hardware, actual collisions) are in play.

  - CRASH_JERK_MPS3: real Normal-class jerk never exceeds ~7.5 m/s³ in this
    data, so 15.0 is also unreachable here. Note this constant was already
    informational only — _check_crash() never gated on it, only recorded it
    in `evidence`. Left as-is; flagging so it isn't mistaken for a tuned,
    load-bearing threshold.

  - HARSH_BRAKE_DECEL_MPS2 (0.4g): fires on ~2.65% of genuine Normal-class
    messages. That rate lines up with vehicles hitting the simulator's
    near-max braking capability at intersections/stop lights — a real
    signal, not a tuning artifact. Left unchanged.

  - HARSH_CORNER_YAW_RATE_DEG_S: the OLD value (25°/s) fired on ~20% of all
    messages and ~5.3% of genuine Normal-class messages — ordinary
    intersection turns at normal city speeds routinely exceed 25°/s, so
    this was flagging routine turning, not harsh swerving. Raised to 90°/s,
    just above the real p99.5 (82°/s) of Normal-class driving, so it now
    reserves "harsh" for genuinely rare, sharp maneuvers.

  - NEAR_MISS: the OLD single-snapshot TTC check fired on ~85% of all
    vehicle pairs within the distance cap. This isn't just a threshold
    problem — naive constant-velocity TTC can't distinguish a genuine
    collision course from two vehicles approaching the same intersection
    that will resolve safely via traffic signals (SUMO vehicles stop; the
    linear TTC projection doesn't know that). The real fix is DEBOUNCING:
    require the SAME vehicle pair to stay under the TTC threshold across
    NEAR_MISS_MIN_CONSECUTIVE consecutive messages from the ego vehicle
    before firing, so a one-frame "will technically intersect if nobody
    brakes" projection doesn't alert on its own. NEAR_MISS_TTC_S was also
    lowered from 2.0s to 1.0s based on the real TTC distribution (median
    TTC among in-range pairs was already under 1s — 2.0s was catching the
    majority of ordinary traffic, not just genuine near-misses).
    This also requires new per-sender state (see `_near_miss_streak`
    below) — if you swap this module out, port that state too.

Re-validate against your own data if the traffic density/geometry differs
meaningfully from LuST/Luxembourg — the debounce window and 90°/s figure are
derived empirically from the dataset above, not first-principles physics,
and will drift with different urban layouts or simulation step sizes.

Usage:
    from driving.kinematic_features import update_kinematics, get_rolling_features, compute_ttc
    from driving.crash_rules import evaluate

    reading = update_kinematics(msg)
    features = get_rolling_features(msg["sender"])
    ttc_info = compute_ttc(msg["sender"])
    event = evaluate(msg["sender"], features, ttc_info)
    if event:
        alerts_log.write(event)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Thresholds — see module docstring above for how each was derived/validated.
# All accel values in m/s^2. 1g ~= 9.81 m/s^2.
# ---------------------------------------------------------------------------

G = 9.81

HARSH_BRAKE_DECEL_MPS2 = 0.4 * G     # ~3.92 m/s^2 — validated against real data, unchanged
CRASH_DECEL_MPS2 = 0.6 * G           # ~5.89 m/s^2 — unreachable in SUMO sim; kept for real hardware
CRASH_POST_SPEED_MPS = 0.6           # ~2 km/h — near-zero speed right after the spike
CRASH_JERK_MPS3 = 15.0               # informational only — not a gating condition, see docstring

NEAR_MISS_TTC_S = 1.0                 # lowered from 2.0s — see docstring
NEAR_MISS_DISTANCE_CAP_M = 60.0       # ignore TTC computed from very far away (noisy bearing math)
NEAR_MISS_MIN_CONSECUTIVE = 2         # NEW — debounce: same pair must stay under TTC this many updates in a row

HARSH_CORNER_YAW_RATE_DEG_S = 90.0    # raised from 25.0 — see docstring (was firing on ordinary turns)

SEVERITY_ORDER = {"HARSH_BRAKE": 1, "HARSH_CORNER": 1, "NEAR_MISS": 2, "CRASH": 3}

# Per-(sender, other_sender) consecutive-near-miss-reading counter, for the
# NEAR_MISS debounce. Cleared lazily — entries for pairs that stop appearing
# simply go stale and get overwritten/ignored; add reset_near_miss_state()
# calls alongside any sender-reset logic in kinematic_features.py if you add
# explicit per-sender resets there (e.g. on the loop/replay guard).
_near_miss_streak: dict = defaultdict(int)


def reset_near_miss_state() -> None:
    _near_miss_streak.clear()


@dataclass
class DrivingEvent:
    sender: str
    timestamp: float
    event_type: str          # "CRASH" | "NEAR_MISS" | "HARSH_BRAKE" | "HARSH_CORNER"
    severity: int             # 1 = warning, 2 = urgent, 3 = critical
    reason: str               # human-readable, feeds directly into the Gemma prompt layer
    evidence: dict             # raw numbers behind the call, for the dashboard/alert card

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

    pair_key = (sender, ttc_info["other_sender"])

    if ttc_info["ttc"] <= NEAR_MISS_TTC_S:
        _near_miss_streak[pair_key] += 1
    else:
        _near_miss_streak[pair_key] = 0
        return None

    if _near_miss_streak[pair_key] < NEAR_MISS_MIN_CONSECUTIVE:
        # Under the TTC threshold, but not yet for enough consecutive
        # readings — likely a transient intersection-approach geometry
        # that will resolve on its own (e.g. the other vehicle stops).
        return None

    return DrivingEvent(
        sender=sender, timestamp=timestamp, event_type="NEAR_MISS", severity=2,
        reason=(
            f"Time-to-collision with vehicle {ttc_info['other_sender']} is "
            f"{ttc_info['ttc']:.1f}s at {ttc_info['distance_m']:.0f}m distance, "
            f"closing at {ttc_info['closing_speed_mps']:.1f} m/s "
            f"(sustained for {_near_miss_streak[pair_key]} consecutive readings)."
        ),
        evidence={**ttc_info, "consecutive_readings": _near_miss_streak[pair_key]},
    )


def evaluate(sender: str, features: Optional[dict], ttc_info: Optional[dict] = None) -> Optional[DrivingEvent]:
    """
    Run all rules for one sender's latest reading and return the single
    highest-severity event, if any (CRASH beats NEAR_MISS beats HARSH_*).

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
