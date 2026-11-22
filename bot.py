import math
import mechfight as mf
from ground_map import GroundMap
from policy import StallRecovery, gun_controls, navigate, resources, select_target


class Aspire:
    name = "ASPIRE"

    def __init__(self, hello: mf.Hello):
        self.config = hello["config"]
        try:
            self.ground_map = GroundMap.load(hello)
        except (OSError, ValueError, KeyError):
            self.ground_map = None
            mf.log("Ground map unavailable; using direct navigation with stall recovery.")
        self.recovery = StallRecovery()

    def lightning_pickup(self, obs, move):
        me = obs["self"]
        if me["weapons"][1] is not None or not me["in_zone"]:
            return None, move
        x, y, z = me["pos"]
        goal = obs["zone"]["next_center"]
        direct = math.dist((x, y), goal)
        reach = self.config["loot"]["pickup_radius"]
        best = None
        for item in obs["pickups"]:
            px, py, pz = item["pos"]
            if item["kind"] != "lightning" or abs(pz-z) > 4:
                continue
            distance = math.hypot(px-x, py-y)
            if distance <= reach:
                return item["id"], move
            if distance > 85 or not me["grounded"] or self.ground_map is None:
                continue
            if obs["mechs"] or obs["projectiles"]:
                continue
            remaining = math.dist((px, py), goal)
            if remaining + me["stats"]["radius"] > obs["zone"]["next_radius"]:
                continue
            extra = distance + remaining - direct
            if extra > 24:
                continue
            terrain = self.ground_map
            if not terrain.clear_rectangle(terrain.cell(x,y), terrain.cell(px,py)):
                continue
            rank = (extra, distance, item["id"])
            if best is None or rank < best[0]:
                best = rank, [(px-x)/distance, (py-y)/distance]
        return None, best[1] if best else move

    def act(self, obs: mf.Observation) -> mf.Action:
        move, boost = navigate(obs)
        if self.ground_map is not None:
            move = self.ground_map.steer(obs, move)
        take, move = self.lightning_pickup(obs, move)
        move = self.recovery.steer(obs, move)
        target = select_target(obs)
        aim, fire, reload = gun_controls(obs, target, self.config)
        pick = resources(obs)

        action: mf.Action = {"tick": obs["tick"], "move": move, "boost": boost}
        if aim is not None:
            action.update(aim=aim, fire=fire, reload=reload)
        elif reload:
            action["reload"] = True
        if take is not None:
            action["take"] = take
        secondary = obs["self"]["weapons"][1]
        if target and aim and secondary and secondary["kind"] == "lightning":
            stats = secondary["stats"]
            me = obs["self"]
            tx, ty, _ = target["pos"]
            x, y, _ = me["pos"]
            yaw_error = math.remainder(math.atan2(ty - y, tx - x) - me["aim"][0], math.tau)
            fire[1] = (
                me["in_zone"] and me["energy"] >= stats["energy"]
                and secondary["cooldown"] <= 0
                and math.dist(me["pos"], target["pos"]) <= stats["range"]
                and abs(yaw_error) <= math.radians(stats["cone_deg"]) / 2
            )
        if pick is not None:
            action["pick"] = pick
        return action


if __name__ == "__main__":
    mf.run(Aspire)
