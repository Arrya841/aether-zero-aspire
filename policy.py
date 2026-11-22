"""Deterministic Step 1 policy contracts and ASPIRE v4 policy exports."""
from __future__ import annotations

import math
from typing import Any, Optional

import mechfight as mf

# Re-export v4 policy components
from combat_policy import (
    AspirePolicy,
    AngularThreatBelief,
    StallRecovery as V4StallRecovery,
    _clamp_vec2,
    _unit_vec2,
    _has_los_3d,
    WEAPON_PRIORITY,
    NUM_BINS,
    BIN_WIDTH,
    BIN_CENTERS,
)


def navigate(obs: mf.Observation) -> tuple[list[float], bool]:
    """Return world-frame movement and grounded emergency boost."""
    me = obs["self"]
    x, y, _ = me["pos"]
    goal_x, goal_y = obs["zone"]["next_center"]
    dx, dy = goal_x - x, goal_y - y
    scale = max(1.0, math.hypot(dx, dy))
    boost = not me["in_zone"] and me["grounded"] and me["energy"] > 0
    return [dx / scale, dy / scale], boost


class StallRecovery:
    """Keep one match's progress sample and a short escape direction."""

    def __init__(self):
        self.anchor = None
        self.sample_time = 0.0
        self.escape_until = 0.0
        self.escape = [0.0, 0.0]
        self.side = 1

    def steer(self, obs: mf.Observation, move: list[float]) -> list[float]:
        me = obs["self"]
        pos = me["pos"][:2]
        now = obs["t"]
        if not me["grounded"] or math.dist(pos, obs["zone"]["next_center"]) < 2:
            self.anchor = None
            self.escape_until = 0.0
            return move
        if now < self.escape_until:
            return self.escape.copy()
        if self.anchor is None:
            self.anchor, self.sample_time = pos, now
        elif now - self.sample_time >= 0.8:
            stalled = math.dist(self.anchor, pos) < 0.75
            self.anchor, self.sample_time = pos, now
            if stalled:
                self.escape = [-self.side * move[1], self.side * move[0]]
                self.side *= -1
                self.escape_until = now + 1.5
                self.anchor = None
                return self.escape.copy()
        return move


def select_target(obs: mf.Observation) -> mf.ActorObs | None:
    """Select the closest tracked opposing mech, breaking ties by id."""
    origin = obs["self"]["pos"]
    return min(
        (actor for actor in obs["mechs"] if actor["kind"] == "mech"),
        key=lambda actor: (math.dist(origin, actor["pos"]), actor["id"]),
        default=None,
    )


def gun_controls(
    obs: mf.Observation, target: mf.ActorObs | None, config: mf.Config
) -> tuple[list[float] | None, list[bool], bool]:
    """Return aim, two slot triggers, and the gun reload request."""
    me = obs["self"]
    gun = me["weapons"][0]
    fire = [False, False]
    if gun is None or gun["kind"] != "gun":
        return None, fire, False

    reload = gun.get("ammo") == 0 and gun.get("reload", 0) == 0
    if target is None:
        return None, fire, reload

    x, y, z = me["pos"]
    tx, ty, tz = target["pos"]
    dx, dy = tx - x, ty - y
    # Observed positions are at the feet. Aim at the base mech center.
    dz = tz + config["mech"]["height"] / 2 - (z + me["stats"]["muzzle_height"])
    # One flight-time estimate using the gun speed reported this match.
    speed = gun["stats"]["speed"]
    travel = min(math.hypot(dx, dy, dz) / speed, 0.45)
    vx, vy, vz = target["vel"]
    dx += vx * travel
    dy += vy * travel
    dz += vz * travel
    horizontal = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, horizontal)
    actual_yaw, actual_pitch = me["aim"]
    yaw_error = math.remainder(yaw - actual_yaw, math.tau)
    aim_error = math.hypot(yaw_error * math.cos(pitch), pitch - actual_pitch)
    distance = math.hypot(horizontal, dz)
    fire[0] = (
        distance <= gun["stats"]["range"]
        and aim_error <= math.atan2(config["mech"]["radius"], distance)
        and gun.get("ammo", 0) > 0
        and gun.get("reload", 0) == 0
    )
    return [yaw, pitch], fire, reload


def resources(obs: mf.Observation) -> int | None:
    """Accept the first upgrade choice when offered."""
    offer = obs.get("offer")
    return 0 if offer and offer["choices"] else None
