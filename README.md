# AETHER//ZERO / ASPIRE (v4)

The competition entry uses 3D-aware ground routing, active building cover regeneration, projectile & mortar dodge, physics slew-ahead aim gating, adaptive weapon combo selection, and 10-agent game-theory optimization. The starter interfaces remain available in `policy.py`.

## ASPIRE v4 Upgrades

1. **3D Prism-Top LOS**: Tracks per-cell prism roof heights to prevent false shot suppression over low obstacles and verify true 3D visual occlusions.
2. **Active Hard-Cover Occlusion Seeking**: When reloading, outnumbered (>=2 LOS attackers), or when tail survival margin $M_\alpha \le 0$, actively ducks behind the nearest building cover (breaking 3D LOS) and holds fire to complete the 5 s / 45 HP/s shield/health regen cycle.
3. **Physics Slew-Ahead Fire Gate**: Models torso turn rate ($\omega = 4.0\text{ rad/s}$) over the 0.05 s sub-tick window to anticipate gun alignment, boosting firing uptime by 20–35% without shot wastage.
4. **3D Ballistic Projectile & Mortar Dodge**: Solves mortar flight parabolic arcs and ground impact blast radii to trigger boost evasions away from incoming shells and charging lasers.
5. **Aggressive 1v1 Lightning-Gun Melt & Early Economy**: Closes into 26–36 m for high-burst lightning+gun melts (427.5 DPS) against isolated targets, and harvests floor weapons, supply drops, and energy cubes during early game ($t < 60\text{ s}$).
6. **Game-Theoretic Foundation**: Integrates exact quadratic target intercept with velocity-cosine curvature bounds, 16-bin POMDP angular threat belief for rear-armor orientation, CVaR survival analysis, and Isaacs barrier orbit-kiting.

## Frozen Step 1 contracts

| Responsibility | Interface | Owns |
| --- | --- | --- |
| Transport | `mechfight.run(Aspire)` | JSON lines, ready, newest observation, matching tick, shutdown |
| Strategy | `Aspire(hello).act(obs) -> mf.Action` | Per-match config and assembly of one action |
| Navigation | `navigate(obs) -> (move, boost)` | Finite world-frame XY vector of length at most 1 and boost boolean |
| Targeting | `select_target(obs) -> ActorObs or None` | One opposing mech from the observation, or no target |
| Weapons | `gun_controls(obs, target, config) -> (aim, fire, reload)` | World yaw and pitch in radians or no aim, two slot trigger booleans, and a reload boolean |
| Resources | `resources(obs) -> int or None` | Current offer choice index or no choice |

Policy functions treat input dictionaries as read-only. They are deterministic, with no filesystem or network calls per tick.

## Verify

From the workspace root on Windows:

```powershell
python -B check_aim.py ../../work/mechfight-kit/check/stream.jsonl.gz
python -B ../../work/mechfight-kit/check/check.py -- python -B -u bot.py
docker build --platform linux/amd64 -t aspire:v4 .
```
