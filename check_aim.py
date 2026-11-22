"""Check moving-target aim using the starter kit's recorded game messages."""
import argparse
import gzip
import json
import math

from bot import Aspire
from policy import gun_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", help="Path to the starter kit's check/stream.jsonl.gz")
    args = parser.parse_args()
    with gzip.open(args.stream, "rt") as stream:
        hello = json.loads(next(stream))
        observation = json.loads(next(stream))

    me = observation["self"]
    me.update(pos=[0, 0, 0], aim=[0, 0], in_zone=True, energy=100)

    def aim(distance, velocity):
        target = {"id": 1, "kind": "mech", "pos": [distance, 0, 0], "vel": [0, velocity, 0]}
        angles, _, _ = gun_controls(observation, target, hello["config"])
        return angles[0]

    assert aim(100, 0) == 0
    assert 0 < aim(100, 12) < 0.1
    assert math.isclose(aim(100, -12), -aim(100, 12))
    assert math.isclose(aim(250, 12), math.atan2(12 * 0.45, 250))

    stats = hello["config"]["weapons"]["lightning"]
    me["weapons"][1] = {"kind": "lightning", "stats": stats, "cooldown": 0}
    angle = math.radians(stats["cone_deg"]) / 2 - 0.01
    observation["mechs"] = [{
        "kind": "mech", "id": 13, "pos": [30 * math.cos(angle), 30 * math.sin(angle), 0],
        "vel": [0, 12, 0],
    }]
    observation["pickups"] = []
    action = Aspire(hello).act(observation)
    assert action["aim"][0] > math.radians(stats["cone_deg"]) / 2
    assert action["fire"][1]
    print("PASS stationary aim, lateral lead, reversal, prediction cap, lightning cone")


if __name__ == "__main__":
    main()
