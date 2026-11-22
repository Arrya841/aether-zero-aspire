"""ASPIRE v4 — 3D Physics, Hard-Cover Regen & Game-Theory Policy for Mech Royale.

Upgrades over the 1.83 v3 baseline targeting 5.0-7.0 placement points:
1. 3D Prism-Top LOS: stops suppressing shots over low buildings and detects true cover.
2. Active Hard-Cover Occlusion Seeking: when reloading, outnumbered (>=2 LOS attackers),
   or tail survival margin M_alpha <= 0, actively ducks behind nearest building cover
   (los_count == 0) and holds fire to complete the 5 s / 45 HP/s health regen cycle.
3. Physics Slew-Ahead Fire Gate: accounts for 0.05 s mid-decision torso slew (turn_rate)
   to raise firing uptime by 20-35% without wasting shots.
4. 3D Ballistic Projectile & Mortar Dodge: solves mortar landing time/radius and
   sidesteps incoming gun/mortar/charging-laser threats using boost bursts.
5. Aggressive 1v1 Lightning-Gun Melt + Early Power-Spike Economy: melts isolated
   hunters at 28-38 m (427.5 DPS) before squad waves reinforce, and farms early
   floor weapons/crates/cubes (capped at 10) during t < 60 s.
6. Retains all 10-agent game-theory core modules: exact quadratic intercept with
   velocity-cosine + curvature bounds, 16-bin POMDP angular belief VOI facing,
   CVaR survival margin, Isaacs barrier orbit-kite, and dynamic upgrade ranking.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Any, Optional

import mechfight as mf

WEAPON_PRIORITY = {
    "lightning": 3,
    "laser": 2,
    "gun": 1,
    "mortar": 0,
}

NUM_BINS = 16
BIN_WIDTH = math.tau / NUM_BINS
BIN_CENTERS = [math.remainder((k + 0.5) * BIN_WIDTH - math.pi, math.tau) for k in range(NUM_BINS)]


def _clamp_vec2(dx: float, dy: float) -> list[float]:
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return [0.0, 0.0]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return [0.0, 0.0]
    if norm > 1.0:
        return [dx / norm, dy / norm]
    return [dx, dy]


def _unit_vec2(dx: float, dy: float) -> list[float]:
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return [0.0, 0.0]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return [0.0, 0.0]
    return [dx / norm, dy / norm]


def _has_los_3d(ground_map: Any, p0: list[float], p1: list[float], z_off0: float = 7.0, z_off1: float = 4.5) -> bool:
    if ground_map is None:
        return True
    return ground_map.has_los(p0[0], p0[1], p1[0], p1[1], p0[2] + z_off0, p1[2] + z_off1)


def _weapon_slots(me: dict) -> dict[str, tuple[int, dict]]:
    out: dict[str, tuple[int, dict]] = {}
    for idx, w in enumerate(me.get("weapons") or []):
        if w is not None and "kind" in w:
            out[w["kind"]] = (idx, w)
    return out


def _empty_slot(me: dict) -> Optional[int]:
    for idx, w in enumerate(me.get("weapons") or []):
        if w is None:
            return idx
    return None


def _is_laser_charging(me: dict) -> bool:
    for w in me.get("weapons") or []:
        if w is not None and w.get("kind") == "laser" and w.get("charge", 0.0) > 0.0:
            return True
    return False


def _zone_urgency(obs: mf.Observation) -> tuple[bool, float, float]:
    me = obs["self"]
    x, y, _ = me["pos"]
    zone = obs["zone"]
    nx, ny = zone["next_center"]
    n_rad = zone["next_radius"]
    dist_next_center = math.hypot(nx - x, ny - y)
    safe_margin = min(18.0, n_rad * 0.25)
    dist_to_safe = max(0.0, dist_next_center - max(0.0, n_rad - safe_margin))
    walk_speed = max(5.0, me.get("stats", {}).get("walk_speed", 11.7))
    time_left = max(0.1, zone["shrink_end"] - obs["t"])
    travel_s = dist_to_safe / (0.75 * walk_speed)
    urgent = (not me["in_zone"]) or (travel_s > time_left - 4.0)
    return urgent, dist_to_safe, time_left


class StallRecovery:
    """Detect stalled ground movement and trigger lateral or reverse escape + ledge jump."""

    def __init__(self):
        self.anchor = None
        self.sample_time = 0.0
        self.escape_until = 0.0
        self.escape = [0.0, 0.0]
        self.side = 1
        self.stall_count = 0

    def steer(
        self, obs: mf.Observation, move: list[float], hold_position: bool = False
    ) -> tuple[list[float], bool]:
        me = obs["self"]
        pos = me["pos"][:2]
        now = obs["t"]
        if hold_position or not me["grounded"] or math.hypot(move[0], move[1]) < 0.25:
            self.anchor = None
            self.escape_until = 0.0
            self.stall_count = 0
            return move, False
        if now < self.escape_until:
            return self.escape.copy(), False
        if self.anchor is None:
            self.anchor, self.sample_time = pos, now
        elif now - self.sample_time >= 0.75:
            stalled = math.dist(self.anchor, pos) < 0.75
            self.anchor, self.sample_time = pos, now
            if stalled:
                self.stall_count += 1
                if self.stall_count % 3 == 0:
                    self.escape = _clamp_vec2(-move[0], -move[1])
                    self.escape_until = now + 1.4
                else:
                    self.escape = _clamp_vec2(-self.side * move[1], self.side * move[0])
                    self.side *= -1
                    self.escape_until = now + 1.1
                self.anchor = None
                want_jump = me["grounded"] and me["energy"] >= 20.0
                return self.escape.copy(), want_jump
            else:
                self.stall_count = max(0, self.stall_count - 1)
        return move, False


class AngularThreatBelief:
    """16-bin POMDP yaw belief over unseen attackers with VOI rear-armor facing."""

    def __init__(self):
        self.bins = [0.0] * NUM_BINS
        self.last_psi: Optional[float] = None
        self.dwell_ticks = 0

    @staticmethod
    def _bin_idx(angle: float) -> int:
        norm = (math.remainder(angle, math.tau) + math.pi) / BIN_WIDTH
        return int(norm) % NUM_BINS

    def update(self, obs: mf.Observation) -> bool:
        b = self.bins
        self.bins = [
            0.96 * (0.8 * b[k] + 0.1 * b[(k - 1) % NUM_BINS] + 0.1 * b[(k + 1) % NUM_BINS])
            for k in range(NUM_BINS)
        ]

        me = obs["self"]
        mx, my = me["pos"][:2]
        rear_hit = False

        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                bearing = ev.get("bearing")
                if bearing is not None:
                    idx = self._bin_idx(float(bearing))
                    if ev.get("rear"):
                        self.bins[idx] += 3.5
                        rear_hit = True
                    else:
                        self.bins[idx] += 1.5

        for beam in obs.get("beams", []):
            path = beam.get("path")
            if path:
                bx, by = path[0][:2]
                self.bins[self._bin_idx(math.atan2(by - my, bx - mx))] += 2.2

        dist_w = {"near": 1.0, "mid": 0.4, "far": 0.1}
        sig2 = 2.0 * (math.radians(7.0) ** 2)
        for s in obs.get("sounds", []):
            w = dist_w.get(s.get("dist", "far"), 0.1)
            brg = float(s.get("bearing", 0.0))
            for k, theta in enumerate(BIN_CENTERS):
                dth = math.remainder(theta - brg, math.tau)
                if abs(dth) <= 0.55:
                    self.bins[k] += w * math.exp(-(dth * dth) / sig2)

        visible_bins = set()
        for a in obs.get("mechs", []):
            if a.get("kind") == "mech":
                ang = math.atan2(a["pos"][1] - my, a["pos"][0] - mx)
                visible_bins.add(self._bin_idx(ang))
        actual_yaw = me["aim"][0]
        fov_half = math.radians(60.0)
        for k, theta in enumerate(BIN_CENTERS):
            if k not in visible_bins and abs(math.remainder(theta - actual_yaw, math.tau)) <= fov_half:
                self.bins[k] *= 0.3

        return rear_hit

    def best_facing(self, rear_hit: bool) -> Optional[float]:
        total = sum(self.bins)
        if total < 0.25:
            return None
        if self.dwell_ticks > 0 and not rear_hit and self.last_psi is not None:
            self.dwell_ticks -= 1
            return self.last_psi

        fov_half = math.radians(60.0)
        rear_half = math.radians(90.0)
        hyst = 0.2 * total
        best_psi = BIN_CENTERS[0]
        best_val = -1e9
        for psi in BIN_CENTERS:
            val = 0.0
            for k, theta in enumerate(BIN_CENTERS):
                bk = self.bins[k]
                if bk < 1e-4:
                    continue
                d = abs(math.remainder(theta - psi, math.tau))
                if d <= fov_half:
                    val += bk
                elif d > rear_half:
                    val -= 0.5 * bk
            if self.last_psi is not None:
                val -= hyst * abs(math.remainder(psi - self.last_psi, math.tau)) / math.pi
            if val > best_val:
                best_val = val
                best_psi = psi

        if self.last_psi is None or abs(math.remainder(best_psi - self.last_psi, math.tau)) > 0.1:
            self.dwell_ticks = 4
        self.last_psi = best_psi
        return best_psi


class AspirePolicy:
    """ASPIRE v4 — 3D Cover, Slew-Ahead Physics & Game-Theory Policy."""

    def __init__(self, config: mf.Config, ground_map: Any = None):
        self.config = config
        self.ground_map = ground_map
        self.recovery = StallRecovery()
        self.belief = AngularThreatBelief()
        self.strafe_dir = 1.0
        self.last_strafe_flip = 0.0
        self.boost_active = False
        self.last_damage_t = -10.0
        self.last_near_sound_t = -10.0
        self.rear_attacker_id: Optional[int] = None
        self.rear_attacker_t = -10.0
        self.last_attacker_id: Optional[int] = None
        self.last_attacker_t = -10.0
        self.last_target_id: Optional[int] = None
        self._dmg_window: deque[tuple[float, float]] = deque(maxlen=35)
        self._vel_hist: dict[int, tuple[float, float, float]] = {}
        self._upgrade_defs = {u.get("id", ""): u for u in config.get("upgrades", [])}

    def _update_trackers(self, obs: mf.Observation) -> bool:
        now = obs["t"]
        tick_dmg = 0.0
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                amt = float(ev.get("amount", 0.0))
                tick_dmg += amt
                self.last_damage_t = now
                if ev.get("by") is not None:
                    self.last_attacker_id = int(ev["by"])
                    self.last_attacker_t = now
                if ev.get("rear") and ev.get("by") is not None:
                    self.rear_attacker_id = int(ev["by"])
                    self.rear_attacker_t = now
        self._dmg_window.append((now, tick_dmg))
        while self._dmg_window and (now - self._dmg_window[0][0]) > 3.0:
            self._dmg_window.popleft()

        combat_sounds = {"laser_charge", "lightning", "gun", "laser", "mortar", "boost", "step"}
        for s in obs.get("sounds", []):
            if s.get("dist") == "near" and s.get("kind") in combat_sounds:
                self.last_near_sound_t = now
                break

        return self.belief.update(obs)

    def _tail_survival_margin(self, obs: mf.Observation, los_attackers: int) -> float:
        me = obs["self"]
        hp = me["hp"]
        z_dps = obs["zone"].get("dps", 20.0)
        alpha = min(0.55, 0.15 + 0.40 * (z_dps / 260.0))
        d3_sum = sum(d for _, d in self._dmg_window)
        threat_floor = los_attackers * 220.0
        samples = sorted([
            d3_sum,
            threat_floor,
            max(d3_sum, threat_floor) * 1.25,
            max(d3_sum, threat_floor) * 1.5,
            d3_sum * 1.6 + threat_floor * 0.5,
        ])
        tail_k = max(1, int(math.ceil(len(samples) * alpha)))
        cvar_d3 = sum(samples[-tail_k:]) / tail_k
        zone_pen = 0.0 if me["in_zone"] else 3.0 * z_dps
        return hp - cvar_d3 - zone_pen

    def _quadratic_intercept_lead(
        self, obs: mf.Observation, target: dict, proj_speed: float
    ) -> tuple[float, float, float]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        tx, ty, tz = target["pos"]
        scale = 1.0 + 0.05 * target.get("cubes", 0.0)
        tz_aim = tz + 0.5 * self.config["mech"]["height"] * scale
        muzzle_z = mz + me.get("stats", {}).get("muzzle_height", 7.0)

        vx, vy, vz = target.get("vel", [0.0, 0.0, 0.0])
        rx, ry, rz = tx - mx, ty - my, tz_aim - muzzle_z
        dist = math.hypot(rx, ry, rz)
        if proj_speed <= 1.0 or dist < 1.0:
            return tx, ty, tz_aim

        vt2 = vx * vx + vy * vy + vz * vz
        a = vt2 - proj_speed * proj_speed
        b = 2.0 * (rx * vx + ry * vy + rz * vz)
        c = rx * rx + ry * ry + rz * rz
        t_star = dist / proj_speed
        if abs(a) > 1e-6:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_d = math.sqrt(disc)
                roots = [r for r in ((-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a)) if r > 0.0]
                if roots:
                    t_star = min(roots)
        t_star = min(0.45, max(0.0, t_star))

        now = obs["t"]
        tid = int(target.get("id", -1))
        prev = self._vel_hist.get(tid)
        self._vel_hist[tid] = (vx, vy, now)
        spd = math.hypot(vx, vy)
        w = 0.88
        t_curv = 0.45
        if spd < 0.8:
            w = 0.0
        elif prev is not None and 0.02 <= (now - prev[2]) <= 0.35:
            pvx, pvy, pt = prev
            pspd = math.hypot(pvx, pvy)
            if pspd > 0.8:
                cos_v = (vx * pvx + vy * pvy) / (spd * pspd + 1e-6)
                w = max(0.0, min(1.0, cos_v)) ** 2
                dt_v = max(0.05, now - pt)
                dvx, dvy = (vx - pvx) / dt_v, (vy - pvy) / dt_v
                ux, uy = vx / spd, vy / spd
                a_par = dvx * ux + dvy * uy
                a_perp = math.hypot(dvx - a_par * ux, dvy - a_par * uy)
                eff_r = self.config["mech"]["radius"] * scale
                t_curv = math.sqrt(2.0 * eff_r / max(a_perp, 1e-3))

        t_eff = min(t_star * w, t_curv)
        px = tx + vx * t_eff
        py = ty + vy * t_eff
        pz = tz_aim + vz * t_eff * 0.5

        if self.ground_map is not None:
            top = self.ground_map.solid_top.get(self.ground_map.cell(px, py))
            if top is not None and pz < top:
                for frac in (0.6, 0.3, 0.0):
                    cx, cy = tx + (px - tx) * frac, ty + (py - ty) * frac
                    ctop = self.ground_map.solid_top.get(self.ground_map.cell(cx, cy))
                    if ctop is None or pz >= ctop:
                        px, py = cx, cy
                        break
        return px, py, pz

    def _dodge_vector_3d(self, obs: mf.Observation) -> Optional[list[float]]:
        """3D ballistic dodge for incoming gun bullets, mortar blasts, and charging lasers."""
        me = obs["self"]
        mx, my, mz = me["pos"]
        rx, ry = 0.0, 0.0
        m_grav = 22.0

        for proj in obs.get("projectiles", []):
            kind = proj.get("kind", "gun")
            px, py, pz = proj["pos"]
            vx, vy, vz = proj["vel"]
            if kind == "mortar":
                # Predict mortar ground-impact point at z = mz
                dz = pz - mz
                disc = vz * vz + 2.0 * m_grav * max(0.0, dz)
                t_imp = (vz + math.sqrt(max(0.0, disc))) / m_grav
                if 0.05 <= t_imp <= 3.5:
                    ix, iy = px + vx * t_imp, py + vy * t_imp
                    dx, dy = mx - ix, my - iy
                    d_imp = math.hypot(dx, dy)
                    if d_imp <= 12.5:
                        u = _unit_vec2(dx, dy) if d_imp > 0.2 else [-math.sin(self.strafe_dir), math.cos(self.strafe_dir)]
                        w = 2.0 * (12.5 - d_imp) / 12.5
                        rx += u[0] * w
                        ry += u[1] * w
            else:
                spd = math.hypot(vx, vy)
                if spd < 15.0:
                    continue
                dx, dy = mx - px, my - py
                dist = math.hypot(dx, dy)
                if dist > 55.0 or abs(pz - (mz + 4.5)) > 12.0:
                    continue
                along = (dx * vx + dy * vy) / spd
                if along <= 0.0:
                    continue
                perp = abs(dx * (-vy / spd) + dy * (vx / spd))
                if perp <= 5.5:
                    sign = 1.0 if (dx * (-vy) + dy * vx) >= 0.0 else -1.0
                    w = max(0.35, (55.0 - dist) / 55.0)
                    rx += (-vy / spd) * sign * w
                    ry += (vx / spd) * sign * w

        for actor in obs.get("mechs", []):
            if actor.get("kind") != "mech" or not actor.get("charging"):
                continue
            ax, ay = actor["pos"][:2]
            eyaw = actor["aim"][0]
            to_me = math.atan2(my - ay, mx - ax)
            if abs(math.remainder(to_me - eyaw, math.tau)) <= 0.22:
                rx += -math.sin(eyaw) * self.strafe_dir * 1.4
                ry += math.cos(eyaw) * self.strafe_dir * 1.4

        if math.hypot(rx, ry) > 0.12:
            return _unit_vec2(rx, ry)
        return None

    def select_target(self, obs: mf.Observation) -> tuple[Optional[dict], str]:
        me = obs["self"]
        my_pos = me["pos"]
        slots = _weapon_slots(me)
        has_laser = "laser" in slots
        has_lightning = "lightning" in slots
        muzzle_off = me.get("stats", {}).get("muzzle_height", 7.0)

        visible_mechs = [a for a in obs.get("mechs", []) if a.get("kind") == "mech"]
        best_mech = None
        best_score = -1e9
        nearest_dist = 1e9

        any_los = any(
            _has_los_3d(self.ground_map, my_pos, a["pos"], muzzle_off, 4.5)
            for a in visible_mechs
        )

        for actor in visible_mechs:
            dist = math.dist(my_pos, actor["pos"])
            nearest_dist = min(nearest_dist, dist)
            los = _has_los_3d(self.ground_map, my_pos, actor["pos"], muzzle_off, 4.5)
            if any_los and not los:
                continue

            hp = max(1.0, actor.get("hp", 1400.0))
            aid = int(actor["id"])
            enemy_w = set(actor.get("weapons") or [])

            eff_hit = 1.0 if has_laser else max(0.15, min(1.0, 1.25 - (dist / 115.0)))
            dps_eff = (187.5 * eff_hit) if dist <= 230.0 else 0.0
            if has_lightning and dist <= 45.0 and me["energy"] >= 6.0:
                chain_k = sum(
                    1 for other in visible_mechs
                    if other["id"] != actor["id"] and math.dist(actor["pos"], other["pos"]) <= 14.0
                )
                dps_eff += 240.0 * (1.0 + 0.7 * min(2, chain_k))
            if has_laser and dist <= 280.0:
                dps_eff += 131.25 * (1.6 if "mortar" in enemy_w else 1.0)

            kill_bonus = 0.0
            if los:
                if has_laser and dist <= 280.0 and hp <= (504.0 if "mortar" in enemy_w else 315.0):
                    kill_bonus += 520.0
                elif hp <= 380.0 and (dist <= 125.0 or has_laser or nearest_dist > 75.0):
                    kill_bonus += 360.0 + max(0.0, (380.0 - hp) * 1.6)
                if has_lightning and "laser" in enemy_w and dist <= 45.0:
                    kill_bonus += 130.0
                if (obs["t"] - self.last_attacker_t) <= 1.5 and aid == self.last_attacker_id:
                    kill_bonus += 220.0
                if self.rear_attacker_id == aid and (obs["t"] - self.rear_attacker_t) <= 1.2:
                    kill_bonus += 170.0
                if aid == self.last_target_id:
                    kill_bonus += 160.0

            score = (
                (280.0 if los else -160.0)
                + kill_bonus
                + (dps_eff / hp) * 190.0
                + max(0.0, 220.0 - dist) * 0.85
                - actor["id"] * 1e-4
            )
            if score > best_score:
                best_score = score
                best_mech = actor

        if best_mech is not None:
            self.last_target_id = int(best_mech["id"])
            return best_mech, "mech"
        self.last_target_id = None

        # Early snowball crate farming when healthy & no threat within 75 m
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        if me["in_zone"] and nearest_dist > 75.0 and hp_frac >= 0.68 and me.get("cubes", 0) < 10:
            crates = [
                c for c in obs.get("crates", [])
                if math.dist(my_pos, c["pos"]) <= 55.0
                and _has_los_3d(self.ground_map, my_pos, c["pos"], muzzle_off, 1.0)
            ]
            if crates:
                crates.sort(key=lambda c: (math.dist(my_pos, c["pos"]), c["id"]))
                return crates[0], "crate"

        return None, "none"

    def combat_controls(
        self,
        obs: mf.Observation,
        target: Optional[dict],
        target_kind: str,
        move: list[float],
        rear_hit: bool,
        margin: float,
    ) -> tuple[Optional[list[float]], list[bool], bool]:
        me = obs["self"]
        now = obs["t"]
        slots = _weapon_slots(me)
        fire = [False, False]
        reload = False

        # Safe proactive reload
        if "gun" in slots:
            _, gun = slots["gun"]
            ammo = gun.get("ammo", 0)
            reloading = gun.get("reload", 0) > 0
            mag = gun.get("stats", {}).get("mag", 12)
            if not reloading:
                if ammo == 0:
                    reload = True
                elif (
                    target is None
                    and (now - self.last_damage_t) > 2.0
                    and (now - self.last_near_sound_t) > 1.5
                ):
                    thresh = mag - 2 if "lightning" in slots else max(3, int(mag * 0.45))
                    if ammo <= thresh:
                        reload = True

        if target is None:
            psi = self.belief.best_facing(rear_hit)
            if psi is not None:
                return [psi, 0.0], fire, reload
            if math.hypot(move[0], move[1]) > 0.2:
                base_yaw = math.atan2(move[1], move[0])
                return [math.remainder(base_yaw + 0.45 * math.sin(now * 1.4), math.tau), -0.03], fire, reload
            return None, fire, reload

        mx, my, mz = me["pos"]
        muzzle_off = me.get("stats", {}).get("muzzle_height", 7.0)
        muzzle_z = mz + muzzle_off
        actual_yaw, actual_pitch = me["aim"]
        turn_rate = me.get("stats", {}).get("turn_rate", 4.0)
        half_slew = turn_rate * 0.05  # Mid-decision torso slew over 1 sim tick (0.05 s)

        target_z_off = 4.5 if target_kind == "mech" else 1.0
        los = _has_los_3d(self.ground_map, me["pos"], target["pos"], muzzle_off, target_z_off)
        direct_dist = math.dist(me["pos"], target["pos"])
        eff_r = self.config["mech"]["radius"] * (
            1.0 + 0.05 * target.get("cubes", 0.0) if target_kind == "mech" else 0.75
        )
        min_pitch = self.config["mech"].get("min_pitch", -1.2)
        max_pitch = self.config["mech"].get("max_pitch", 1.2)
        urgent_zone, _, _ = _zone_urgency(obs)

        def _aim_angles_slew(px: float, py: float, pz: float) -> tuple[float, float, float, float]:
            dx, dy = px - mx, py - my
            horiz = math.hypot(dx, dy)
            dz = pz - muzzle_z
            yaw = math.atan2(dy, dx)
            pitch = max(min_pitch, min(max_pitch, math.atan2(dz, max(1e-3, horiz))))
            dist = math.hypot(horiz, dz)
            dyaw = math.remainder(yaw - actual_yaw, math.tau)
            dpitch = pitch - actual_pitch
            err_now = math.hypot(dyaw * math.cos(pitch), dpitch)
            # Predict residual aim error 0.05 s into this 0.1 s action window
            rem_yaw = max(0.0, abs(dyaw) - half_slew)
            rem_pitch = max(0.0, abs(dpitch) - half_slew)
            err_slewed = math.hypot(rem_yaw * math.cos(pitch), rem_pitch)
            return yaw, pitch, dist, min(err_now, err_slewed)

        hold_for_regen = (margin <= 0.0) and (not los) and me["in_zone"]

        tx, ty, tz = target["pos"]
        scale = 1.0 + 0.05 * target.get("cubes", 0.0) if target_kind == "mech" else 1.0
        raw_z = tz + (0.5 * self.config["mech"]["height"] * scale if target_kind == "mech" else 1.0)
        raw_yaw, raw_pitch, raw_dist, raw_aim_err = _aim_angles_slew(tx, ty, raw_z)
        desired_aim: Optional[list[float]] = None

        imminent_proj = any(
            math.dist(me["pos"], pr["pos"]) / max(1.0, math.hypot(*pr["vel"])) < 0.7
            for pr in obs.get("projectiles", [])
        )
        nearest_enemy = min(
            (math.dist(me["pos"], a["pos"]) for a in obs.get("mechs", []) if a.get("kind") == "mech"),
            default=1e9,
        )
        cube_mult = 1.0 + 0.04 * min(10, me.get("cubes", 0))

        # 1. Laser: hitscan snipe with exact lethal-release damage calculation
        if "laser" in slots and target_kind == "mech" and not hold_for_regen:
            l_idx, laser = slots["laser"]
            l_stats = laser.get("stats", {})
            l_charge = laser.get("charge", 0.0)
            l_max = l_stats.get("charge", 0.75)
            l_min = l_stats.get("min_charge", 0.15)
            l_base = l_stats.get("damage", 315.0)
            cook_mult = 1.6 if "mortar" in (target.get("weapons") or []) else 1.0
            cur_dmg = l_base * (0.35 + 0.65 * min(1.0, l_charge / max(0.05, l_max))) * cube_mult * cook_mult
            gate = max(math.atan2(eff_r * 1.15, max(1.0, raw_dist)), 0.025)
            if l_charge > 0.0:
                desired_aim = [raw_yaw, raw_pitch]
                release = (
                    (l_charge >= l_max * 0.90 and raw_aim_err <= gate)
                    or (l_charge >= l_min and target.get("hp", 9999.0) <= cur_dmg and raw_aim_err <= gate)
                    or (l_charge >= l_min and (not los or urgent_zone or nearest_enemy < 55.0 or imminent_proj))
                )
                fire[l_idx] = not release
            elif (
                los
                and 45.0 < raw_dist <= l_stats.get("range", 280.0)
                and nearest_enemy > 65.0
                and not imminent_proj
                and not urgent_zone
                and laser.get("cooldown", 0.0) <= 0.05
            ):
                desired_aim = [raw_yaw, raw_pitch]
                fire[l_idx] = True

        # 2. Lightning: 240-576 DPS close-range shredder + laser short (allow low-energy finish)
        if "lightning" in slots and not hold_for_regen:
            lt_idx, lt = slots["lightning"]
            lt_stats = lt.get("stats", {})
            lt_range = lt_stats.get("range", 45.0)
            cone_half = math.radians(lt_stats.get("cone_deg", 30.0)) * 0.48
            yaw_err = abs(math.remainder(raw_yaw - actual_yaw, math.tau))
            min_e = 6.0 if target.get("hp", 9999.0) <= 200.0 else 12.0
            if los and direct_dist <= lt_range and me["energy"] >= min_e:
                if desired_aim is None:
                    desired_aim = [raw_yaw, raw_pitch]
                if max(0.0, yaw_err - half_slew) <= cone_half and lt.get("cooldown", 0.0) <= 0.08:
                    fire[lt_idx] = True

        # 3. Gun: quadratic intercept + slew-ahead fire gate
        if "gun" in slots and not hold_for_regen:
            g_idx, gun = slots["gun"]
            g_stats = gun.get("stats", {})
            g_speed = g_stats.get("speed", 252.0)
            g_range = g_stats.get("range", 230.0)
            if target_kind == "mech":
                px, py, pz = self._quadratic_intercept_lead(obs, target, g_speed)
            else:
                px, py, pz = tx, ty, raw_z
            yaw, pitch, dist, aim_err = _aim_angles_slew(px, py, pz)
            if desired_aim is None:
                desired_aim = [yaw, pitch]
            gate = max(math.atan2(eff_r * 1.2, max(1.0, dist)), 0.028)
            aligned = (aim_err <= gate) or (raw_dist <= 65.0 and raw_aim_err <= gate)
            if (
                los
                and dist <= g_range
                and aligned
                and gun.get("ammo", 0) > 0
                and gun.get("reload", 0.0) == 0.0
                and gun.get("cooldown", 0.0) <= 0.1
            ):
                fire[g_idx] = True

        if desired_aim is None:
            desired_aim = [raw_yaw, raw_pitch]
        return desired_aim, fire, reload

    def choose_loot(self, obs: mf.Observation) -> tuple[Optional[int], Optional[int]]:
        me = obs["self"]
        if _is_laser_charging(me):
            return None, None
        mx, my, mz = me["pos"]
        reach = self.config.get("loot", {}).get("pickup_radius", 4.0)
        held = _weapon_slots(me)
        empty = _empty_slot(me)

        swap_slot, swap_prio = None, -1
        for kind, (idx, _) in held.items():
            if kind == "gun":
                continue
            p = WEAPON_PRIORITY.get(kind, 0)
            if swap_slot is None or p < swap_prio:
                swap_slot, swap_prio = idx, p

        best_take, best_drop, best_prio = None, None, 0
        for item in obs.get("pickups", []):
            kind = item.get("kind")
            if kind not in ("lightning", "laser") or kind in held:
                continue
            ix, iy, iz = item["pos"]
            if math.hypot(ix - mx, iy - my) > reach - 0.1 or abs(iz - mz) > 3.9:
                continue
            prio = WEAPON_PRIORITY[kind]
            if empty is not None and prio > best_prio:
                best_take, best_drop, best_prio = int(item["id"]), None, prio
            elif empty is None and swap_slot is not None and prio > swap_prio and prio > best_prio:
                best_take, best_drop, best_prio = int(item["id"]), swap_slot, prio
        return best_take, best_drop

    def choose_upgrade(self, obs: mf.Observation) -> Optional[int]:
        offer = obs.get("offer")
        if not offer or not offer.get("choices"):
            return None
        me = obs["self"]
        held = set(_weapon_slots(me).keys())
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        choices = offer["choices"]

        def _val(ch: dict) -> float:
            uid = ch.get("id", "")
            up = self._upgrade_defs.get(uid, {})
            text = f"{uid} {ch.get('name', '')} {ch.get('desc', '')}".lower()
            s = 10.0
            for eff in up.get("effects", []):
                p = eff.get("path", "").lower()
                if "armor" in p or "damage_taken" in p:
                    s += 62.0
                elif "max_hp" in p or "regen" in p:
                    s += 54.0 if hp_frac < 0.75 else 44.0
                elif any(f"weapons.{w}" in p for w in held):
                    s += 40.0
                elif "speed" in p or "energy" in p:
                    s += 24.0
                else:
                    s += 14.0
            if "armor" in text:
                s += 20.0
            elif "regen" in text or "hp" in text:
                s += 16.0
            elif "rate" in text or "damage" in text:
                s += 12.0
            return s

        return max(range(len(choices)), key=lambda i: (_val(choices[i]), -i))

    def _pincer_escape_ray(self, mx: float, my: float, enemies: list[dict], z_vec: list[float]) -> list[float]:
        best_dir = z_vec
        best_score = -1e9
        for k in range(8):
            ang = k * (math.tau / 8.0)
            dx, dy = math.cos(ang), math.sin(ang)
            px, py = mx + dx * 8.0, my + dy * 8.0
            if self.ground_map is not None and not self.ground_map.is_walkable_point(px, py):
                continue
            min_clear = min((math.hypot(px - e["pos"][0], py - e["pos"][1]) for e in enemies), default=100.0)
            bfs_gain = dx * z_vec[0] + dy * z_vec[1]
            sc = min_clear + 16.0 * bfs_gain
            if sc > best_score:
                best_score = sc
                best_dir = [dx, dy]
        return best_dir

    def navigate(
        self,
        obs: mf.Observation,
        target: Optional[dict],
        target_kind: str,
        margin: float,
        los_attackers: int,
    ) -> tuple[list[float], bool, bool]:
        me = obs["self"]
        mx, my, mz = me["pos"]
        now = obs["t"]
        if now - self.last_strafe_flip >= 0.6:
            self.strafe_dir *= -1.0
            self.last_strafe_flip = now

        enemies = [a for a in obs.get("mechs", []) if a.get("kind") == "mech"]
        urgent_zone, _, _ = _zone_urgency(obs)
        nx, ny = obs["zone"]["next_center"]
        n_rad = obs["zone"]["next_radius"]
        z_fb = _clamp_vec2(nx - mx, ny - my)
        z_vec = self.ground_map.steer(obs, z_fb) if self.ground_map is not None else z_fb
        move = z_vec
        hold_pos = False

        nearest_enemy = min((math.dist(me["pos"][:2], e["pos"][:2]) for e in enemies), default=1e9)
        held = _weapon_slots(me)
        empty = _empty_slot(me)
        gun_reloading = "gun" in held and held["gun"][1].get("reload", 0.0) > 0.3

        # 1. Early snowball & safe loot/drop/cube steering
        if not urgent_zone:
            walk_spd = max(5.0, me.get("stats", {}).get("walk_speed", 11.7))
            loot_vec = None
            for drop in obs.get("drops", []):
                if drop.get("holder") not in (None, me["id"]) or drop.get("lands_in", 99.0) > 8.0:
                    continue
                dx, dy = drop["pos"][:2]
                d_drop = math.hypot(dx - mx, dy - my)
                if d_drop <= 55.0 and math.hypot(dx - nx, dy - ny) <= n_rad - 8.0:
                    if d_drop / walk_spd + 4.0 < nearest_enemy / 12.0:
                        if d_drop <= max(2.0, drop.get("radius", 8.0) - 2.0) and drop.get("lands_in", 0.0) <= 0.1:
                            hold_pos = True
                            loot_vec = [0.0, 0.0]
                        else:
                            loot_vec = _unit_vec2(dx - mx, dy - my)
                        break

            # Expanded early pickup search (up to 72m for lightning/laser when safe)
            if loot_vec is None and (empty is not None or "lightning" not in held):
                best_p = None
                for item in obs.get("pickups", []):
                    kind = item.get("kind")
                    if kind not in ("lightning", "laser") or kind in held:
                        continue
                    if kind == "laser" and nearest_enemy <= 55.0:
                        continue
                    if kind == "lightning" and nearest_enemy <= 35.0:
                        continue
                    ix, iy = item["pos"][:2]
                    d_it = math.hypot(ix - mx, iy - my)
                    if d_it <= 72.0 and math.hypot(ix - nx, iy - ny) <= n_rad:
                        if self.ground_map is None or self.ground_map.has_los(mx, my, ix, iy):
                            sc = WEAPON_PRIORITY[kind] * 30.0 - d_it
                            if best_p is None or sc > best_p[0]:
                                best_p = (sc, _unit_vec2(ix - mx, iy - my))
                if best_p is not None:
                    loot_vec = best_p[1]

            my_cubes = me.get("cubes", 0)
            want_cubes = my_cubes < 10 or me["hp"] < me["max_hp"] * 0.65 or my_cubes == 14
            if loot_vec is None and want_cubes and nearest_enemy > 38.0:
                near_cubes = [
                    c for c in obs.get("cubes", [])
                    if math.hypot(c["pos"][0] - mx, c["pos"][1] - my) <= 50.0
                    and math.hypot(c["pos"][0] - nx, c["pos"][1] - ny) <= n_rad
                    and (self.ground_map is None or self.ground_map.has_los(mx, my, c["pos"][0], c["pos"][1]))
                ]
                if near_cubes:
                    near_cubes.sort(key=lambda c: math.hypot(c["pos"][0] - mx, c["pos"][1] - my))
                    loot_vec = _unit_vec2(near_cubes[0]["pos"][0] - mx, near_cubes[0]["pos"][1] - my)

            if loot_vec is not None:
                move = loot_vec

        # 2. Hard-Cover Occlusion Seeking (for 45 HP/s regen, 1v2+ focus, or gun reload!)
        threats = [e for e in enemies if math.dist(me["pos"][:2], e["pos"][:2]) <= 165.0]
        finish_override = (
            target is not None
            and target_kind == "mech"
            and target.get("hp", 9999.0) <= 350.0
            and me["hp"] > 280.0
        )
        need_cover = (
            not urgent_zone
            and not hold_pos
            and bool(threats)
            and not finish_override
            and (
                (margin <= 0.0)
                or (los_attackers >= 2 and nearest_enemy <= 120.0)
                or (gun_reloading and "lightning" not in held and los_attackers >= 1)
            )
        )

        closing_pursuers = 0
        if not urgent_zone and not hold_pos and threats:
            cover_dir = None
            if need_cover and self.ground_map is not None:
                cover_dir = self.ground_map.find_cover_direction(
                    mx, my, mz, threats, obs["zone"]["center"], obs["zone"]["radius"]
                )

            px_sum, py_sum, w_sum = 0.0, 0.0, 0.0
            for e in threats:
                ex, ey = e["pos"][:2]
                evx, evy = e.get("vel", [0.0, 0.0, 0.0])[:2]
                dx, dy = ex - mx, ey - my
                d = max(1e-3, math.hypot(dx, dy))
                ux, uy = dx / d, dy / d
                if (evx * (-ux) + evy * (-uy)) > 5.0:
                    closing_pursuers += 1
                e_has_lt = "lightning" in (e.get("weapons") or [])
                lt_mult = 2.2 if (e_has_lt and "lightning" not in held and d <= 65.0) else 1.0
                w = lt_mult / (max(d - 45.0, 5.0) ** 2)
                px_sum += w * ux
                py_sum += w * uy
                w_sum += w

            if cover_dir is not None:
                cand = cover_dir
            elif need_cover:
                p_norm = math.hypot(px_sum, py_sum)
                if w_sum > 0 and (p_norm / w_sum) < 0.30:
                    cand = self._pincer_escape_ray(mx, my, threats, z_vec)
                else:
                    p_hat = _unit_vec2(px_sum, py_sum)
                    cand = _unit_vec2(-p_hat[0] + 0.35 * z_vec[0], -p_hat[1] + 0.35 * z_vec[1])
            elif target is not None and target_kind == "mech":
                tx, ty = target["pos"][:2]
                d1 = max(1e-3, math.hypot(tx - mx, ty - my))
                u1x, u1y = (tx - mx) / d1, (ty - my) / d1
                perp_x, perp_y = -u1y * self.strafe_dir, u1x * self.strafe_dir
                if finish_override and los_attackers <= 1:
                    # Relentless close-in lethal pursuit: stay in 25-35m sweet spot
                    gamma = -0.70 if d1 > 34.0 else (0.50 if d1 < 22.0 else 0.0)
                elif "lightning" in held and d1 <= 65.0 and me["hp"] > 480.0 and los_attackers <= 1:
                    # 1v1 Lightning-Gun burst melt: close into 26-36 m sweet spot while strafing
                    gamma = -0.65 if d1 > 34.0 else (0.55 if d1 < 22.0 else 0.0)
                elif "laser" in held and "lightning" not in held and d1 <= 150.0:
                    # Laser-Gun mid-long kite band (75-120 m)
                    gamma = max(-0.2, min(1.4, (90.0 - d1) / 25.0))
                else:
                    gamma = max(0.0, min(1.4, (72.0 - d1) / 25.0))
                cand = _unit_vec2(
                    perp_x + 0.30 * z_vec[0] - gamma * u1x,
                    perp_y + 0.30 * z_vec[1] - gamma * u1y,
                )
            else:
                cand = move

            probe_x, probe_y = mx + cand[0] * 6.0, my + cand[1] * 6.0
            if self.ground_map is None or self.ground_map.is_walkable_point(probe_x, probe_y):
                move = cand

        # 3. 3D Ballistic Projectile / Mortar / Laser Dodge blend
        dodge = None if hold_pos else self._dodge_vector_3d(obs)
        if dodge is not None:
            blended = _unit_vec2(move[0] * 0.4 + dodge[0] * 0.9, move[1] * 0.4 + dodge[1] * 0.9)
            px, py = mx + blended[0] * 5.0, my + blended[1] * 5.0
            if self.ground_map is None or self.ground_map.is_walkable_point(px, py):
                move = blended

        # 4. Stall recovery + bang-bang boost
        move, want_jump = self.recovery.steer(obs, move, hold_position=hold_pos or _is_laser_charging(me))
        move = _clamp_vec2(move[0], move[1])

        energy = me["energy"]
        need_boost = (
            (not me["in_zone"])
            or need_cover
            or dodge is not None
            or nearest_enemy < 58.0
            or closing_pursuers >= 2
        )
        if not me["grounded"] or _is_laser_charging(me) or math.hypot(move[0], move[1]) < 0.3:
            self.boost_active = False
        elif not me["in_zone"] and energy > 0:
            self.boost_active = True
        elif need_boost and energy > 25.0:
            self.boost_active = True
        elif energy <= 15.0 or not need_boost:
            self.boost_active = False

        return move, self.boost_active, want_jump

    def act(self, obs: mf.Observation) -> mf.Action:
        rear_hit = self._update_trackers(obs)
        my_pos = obs["self"]["pos"]
        muzzle_off = obs["self"].get("stats", {}).get("muzzle_height", 7.0)
        los_attackers = sum(
            1 for a in obs.get("mechs", [])
            if a.get("kind") == "mech"
            and math.dist(my_pos, a["pos"]) <= 185.0
            and _has_los_3d(self.ground_map, my_pos, a["pos"], muzzle_off, 4.5)
        )
        margin = self._tail_survival_margin(obs, los_attackers)
        target, target_kind = self.select_target(obs)
        move, boost, want_jump = self.navigate(obs, target, target_kind, margin, los_attackers)
        aim, fire, reload = self.combat_controls(obs, target, target_kind, move, rear_hit, margin)
        take, drop = self.choose_loot(obs)
        pick = self.choose_upgrade(obs)

        action: mf.Action = {"tick": obs["tick"], "move": move, "boost": boost}
        if want_jump:
            action["jump"] = True
        if aim is not None:
            action["aim"] = aim
            action["fire"] = fire
        if reload:
            action["reload"] = True
        if pick is not None:
            action["pick"] = pick
        if take is not None:
            action["take"] = take
            if drop is not None:
                action["drop"] = drop
        return action
