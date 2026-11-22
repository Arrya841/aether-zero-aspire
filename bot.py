import mechfight as mf
from ground_map import GroundMap
from combat_policy import AspirePolicy


class Aspire:
    name = "ASPIRE"

    def __init__(self, hello: mf.Hello):
        self.config = hello["config"]
        try:
            self.ground_map = GroundMap.load(hello)
        except (OSError, ValueError, KeyError):
            self.ground_map = None
            mf.log("Ground map unavailable; using direct navigation with stall recovery.")
        self.policy = AspirePolicy(self.config, self.ground_map)

    def _init_with_map(self, hello: mf.Hello, ground_map: GroundMap | None) -> None:
        self.config = hello["config"]
        self.ground_map = ground_map
        self.policy = AspirePolicy(self.config, self.ground_map)

    def act(self, obs: mf.Observation) -> mf.Action:
        return self.policy.act(obs)


if __name__ == "__main__":
    mf.run(Aspire)
