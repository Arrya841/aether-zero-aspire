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

    def act(self, obs: mf.Observation) -> mf.Action:
        move, boost = navigate(obs)
        if self.ground_map is not None:
            move = self.ground_map.steer(obs, move)
        move = self.recovery.steer(obs, move)
        target = select_target(obs)
        aim, fire, reload = gun_controls(obs, target, self.config)
        pick = resources(obs)

        action: mf.Action = {"tick": obs["tick"], "move": move, "boost": boost}
        if aim is not None:
            action.update(aim=aim, fire=fire, reload=reload)
        elif reload:
            action["reload"] = True
        if pick is not None:
            action["pick"] = pick
        return action


if __name__ == "__main__":
    mf.run(Aspire)
