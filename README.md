# AETHER//ZERO / ASPIRE

Step 1 is a deterministic Mech Royale baseline. It walks toward the next zone center, boosts on the ground outside the current circle, aims at the nearest tracked opposing mech, fires the starting gun when aim and range allow, and accepts the first upgrade choice.

## Frozen Step 1 contracts

| Responsibility | Interface | Owns |
| --- | --- | --- |
| Transport | `mechfight.run(Aspire)` | JSON lines, ready, newest observation, matching tick, shutdown |
| Strategy | `Aspire(hello).act(obs) -> mf.Action` | Per-match config and assembly of one action |
| Navigation | `navigate(obs) -> (move, boost)` | Finite world-frame XY vector of length at most 1 and boost boolean |
| Targeting | `select_target(obs) -> ActorObs or None` | One opposing mech from the observation, or no target |
| Weapons | `gun_controls(obs, target, config) -> (aim, fire, reload)` | World yaw and pitch in radians or no aim, two slot trigger booleans, and a reload boolean |
| Resources | `resources(obs) -> int or None` | Current offer choice index or no choice |

Policy functions treat input dictionaries as read-only. They are deterministic, with no filesystem or network calls per tick. The strategy owns the merge and the original observation tick. An absent target suppresses fire and keeps the previous commanded aim. Slot 1 stays idle. Keep protocol I/O in the wrapper. Future map initialization belongs before ready and should use the map path in hello.

## Verify

From the workspace root on Windows:

```powershell
python -B work/mechfight-kit/check/check.py -- python -B -u outputs/aspire/bot.py
docker build --platform linux/amd64 -t aspire:step1 outputs/aspire
```

With the workspace mounted on a Linux shell and Docker running:

```sh
python3 -B work/mechfight-kit/check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --memory-swap 512m --pids-limit 128 --cap-drop ALL --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/work/mechfight-kit/arena:/arena:ro" aspire:step1
```

Run the container check from Linux or WSL so the official checker preserves the Linux path /arena in hello. The fixture checks protocol behavior and timing. Practice matches measure placement.

## Submit

Use this folder as the public GitHub repository root, with the Dockerfile at the top. The official limit is 20 MB at the submitted commit. Submit the repository URL on the Mech Royale track page after the container check passes.

The Step 1 Docker check on Windows used the official checker through a local launcher that preserves the Linux map path in hello:

```powershell
python -B work/check_docker_windows.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --memory-swap 512m --pids-limit 128 --cap-drop ALL --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "${PWD}/work/mechfight-kit/arena:/arena:ro" aspire:step1
```
The development version adds per-match stall recovery after navigation. When grounded movement covers less than 0.75 m over 0.8 s away from the goal, it steps sideways for 1.5 s and retries the zone route. Repeated stalls alternate sides.
Map navigation reads v2/world.json during the handshake. A four-metre grid routes around buildings and stadium footprints with a clearance margin. A bounded search spreads work across observations. If the map is unavailable, the bot uses direct navigation and stall recovery. The ground planner omits bridge decks, terrain slopes, and vertical traversal.

## Research-based navigation

Free-space steering skips up to five extra waypoints after checking the clearance grid. Near opponents the bot ranks equally short next steps by the minimum separation to any observed opposing mech, assuming it either holds position or continues moving for half a second. These are independent standard-library implementations inspired by [PathFinding.js path smoothing](https://github.com/qiao/PathFinding.js/blob/master/src/core/Util.js) (MIT) and [OpenSpiel adversarial search](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/minimax.py) (Apache-2.0). The movement score uses horizontal distance and does not model cover, damage or opponent intent.
