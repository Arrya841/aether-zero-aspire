"""Ground routing, line-of-sight, and optional terrain queries on a 4 m grid."""
from array import array
from collections import deque
import json
import math
from pathlib import Path

STEP = 4


class GroundMap:
    def __init__(self, world, radius, terrain_path=None):
        bounds = world["bounds"]
        self.x0, self.y0 = bounds["minX"], bounds["minY"]
        self.width = math.ceil((bounds["maxX"] - self.x0) / STEP)
        self.height = math.ceil((bounds["maxY"] - self.y0) / STEP)
        self.solid = set()
        for prism in world.get("prisms", []):
            if prism.get("kind") == "bridge":
                continue
            solid = self.fill(prism["ring"])
            for hole in prism.get("holes", []):
                solid.difference_update(self.fill(hole))
            self.solid.update(solid)

        # Dilate once after all prisms are rasterized for mech radius + grid error.
        padding = math.ceil(radius / STEP)
        self.blocked = {
            (x + dx, y + dy)
            for x, y in self.solid
            for dx in range(-padding, padding + 1)
            for dy in range(-padding, padding + 1)
            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
        }

        # Optional 1 m terrain heightmap (v2/terrain.i16) for cliff/slope gating.
        self._terrain = None
        self._t_meta = None
        if terrain_path is not None and "terrain" in world:
            self._load_terrain(terrain_path, world["terrain"])

        self.goal = None
        self.frontier = deque()
        self.next_cell = {}
        self.depth = {}

    def _load_terrain(self, path: Path, meta: dict) -> None:
        try:
            nx = int(meta["nx"])
            ny = int(meta["ny"])
            raw = path.read_bytes()
            if len(raw) < nx * ny * 2:
                return
            arr = array("h")
            arr.frombytes(raw[: nx * ny * 2])
            self._terrain = arr
            self._t_meta = (
                float(meta["minX"]),
                float(meta["minY"]),
                float(meta.get("step", 1.0)),
                nx,
                ny,
            )
        except (OSError, KeyError, ValueError, TypeError):
            self._terrain = None
            self._t_meta = None

    @classmethod
    def load(cls, hello):
        map_dir = Path(hello["map"]["dir"])
        world_path = map_dir / "v2" / "world.json"
        terrain_path = map_dir / "v2" / "terrain.i16"
        with world_path.open(encoding="utf-8") as stream:
            world = json.load(stream)
        return cls(
            world,
            hello["config"]["mech"]["radius"],
            terrain_path=terrain_path if terrain_path.is_file() else None,
        )

    def ground_z(self, x: float, y: float) -> float | None:
        if self._terrain is None or self._t_meta is None:
            return None
        min_x, min_y, step, nx, ny = self._t_meta
        fx = (x - min_x) / step
        fy = (y - min_y) / step
        ix = min(max(int(math.floor(fx)), 0), nx - 2)
        iy = min(max(int(math.floor(fy)), 0), ny - 2)
        tx = min(max(fx - ix, 0.0), 1.0)
        ty = min(max(fy - iy, 0.0), 1.0)
        row0 = iy * nx
        row1 = row0 + nx
        z00 = self._terrain[row0 + ix]
        z10 = self._terrain[row0 + ix + 1]
        z01 = self._terrain[row1 + ix]
        z11 = self._terrain[row1 + ix + 1]
        z0 = z00 + (z10 - z00) * tx
        z1 = z01 + (z11 - z01) * tx
        return (z0 + (z1 - z0) * ty) * 0.01

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return math.floor((x - self.x0) / STEP), math.floor((y - self.y0) / STEP)

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        return self.x0 + (cell[0] + 0.5) * STEP, self.y0 + (cell[1] + 0.5) * STEP

    def fill(self, ring):
        vertices = list(zip(ring[::2], ring[1::2]))
        if len(vertices) < 3:
            return set()
        edges = list(zip(vertices, vertices[1:] + vertices[:1]))
        rows = [self.cell(x, y)[1] for x, y in vertices]
        cells = set()
        for row in range(min(rows), max(rows) + 1):
            y = self.y0 + (row + 0.5) * STEP
            crosses = sorted(
                ax + (y - ay) * (bx - ax) / (by - ay)
                for (ax, ay), (bx, by) in edges
                if (ay > y) != (by > y)
            )
            for left, right in zip(crosses[::2], crosses[1::2]):
                for col in range(self.cell(left, y)[0], self.cell(right, y)[0] + 1):
                    cells.add((col, row))
        return cells

    def open(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.blocked

    def is_walkable_point(self, x: float, y: float) -> bool:
        return self.open(self.cell(x, y))

    def has_los(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Fast 2D grid line-of-sight check against solid building footprints."""
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < STEP:
            return True
        steps = max(2, math.ceil(dist / (STEP * 0.5)))
        dx = (x1 - x0) / steps
        dy = (y1 - y0) / steps
        # Skip the very first and last sample so endpoints hugging a wall don't self-block.
        for i in range(1, steps):
            if self.cell(x0 + dx * i, y0 + dy * i) in self.solid:
                return False
        return True

    def nearest_open(self, cell: tuple[int, int]):
        x, y = cell
        for r in range(1, 32):
            edge = [(x + d, y + s * r) for d in range(-r, r + 1) for s in (-1, 1)]
            edge.extend((x + s * r, y + d) for d in range(1 - r, r) for s in (-1, 1))
            candidates = [p for p in edge if self.open(p)]
            if candidates:
                return min(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        return None

    def _passable_step(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        if not self.open(b):
            return False
        if self._terrain is None:
            return True
        za = self.ground_z(*self.cell_center(a))
        zb = self.ground_z(*self.cell_center(b))
        if za is None or zb is None:
            return True
        # Walkable slope limit is 50 deg (~1.19 * 4m = 4.76m per cell); allow slight margin.
        return abs(zb - za) <= 4.6

    def steer(self, obs, fallback, goal_xy=None, avoid_enemies=True):
        """Route toward goal_xy (defaulting to zone.next_center) on the 4 m grid."""
        x, y, _ = obs["self"]["pos"]
        gx, gy = goal_xy if goal_xy is not None else obs["zone"]["next_center"]
        start = self.cell(x, y)
        goal = self.cell(gx, gy)
        if self.goal != goal:
            self.goal = goal
            dest = goal if self.open(goal) else self.nearest_open(goal)
            self.next_cell.clear()
            self.depth.clear()
            self.frontier.clear()
            if dest is not None:
                self.frontier.append(dest)
                self.next_cell[dest] = dest
                self.depth[dest] = 0

        distance = math.hypot(gx - x, gy - y)
        if distance < 16:
            steps = max(1, math.ceil(distance))
            if all(
                self.cell(x + (gx - x) * i / steps, y + (gy - y) * i / steps) not in self.solid
                for i in range(steps + 1)
            ):
                return fallback
        if start in self.solid:
            return fallback
        if not self.open(start):
            exit_cell = self.nearest_open(start)
            if exit_cell is not None:
                return self.direction(x, y, exit_cell)
            return fallback

        # Incrementally expand reverse shortest-path field from goal.
        for _ in range(1400):
            if start in self.next_cell or not self.frontier:
                break
            a, b = here = self.frontier.popleft()
            for neighbor in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if neighbor not in self.next_cell and self._passable_step(here, neighbor):
                    self.next_cell[neighbor] = here
                    self.depth[neighbor] = self.depth[here] + 1
                    self.frontier.append(neighbor)

        dest = self.next_cell.get(start)
        if dest is None or (dest == start and start == goal):
            return fallback

        if avoid_enemies:
            enemies = [
                actor
                for actor in obs.get("mechs", [])
                if actor.get("kind") == "mech"
                and math.dist(actor["pos"][:2], (x, y)) <= 80
            ]
            if enemies and dest != start and obs["self"]["in_zone"]:
                dest = self.choose_step(start, dest, enemies)
                return self.direction(x, y, dest)

        # Skip up to five waypoints if their bounding rectangle is completely open.
        for _ in range(5):
            farther = self.next_cell[dest]
            if farther == dest or not self.clear_rectangle(start, farther):
                break
            dest = farther
        return self.direction(x, y, dest)

    def clear_rectangle(self, start, end):
        left, right = sorted((start[0], end[0]))
        bottom, top = sorted((start[1], end[1]))
        return all(
            self.open((col, row))
            for col in range(left, right + 1)
            for row in range(bottom, top + 1)
        )

    def choose_step(self, start, default, enemies):
        """Maximize worst-case horizontal separation among equal-length routes."""
        col, row = start
        candidates = [default]
        for cell in ((col + 1, row), (col - 1, row), (col, row + 1), (col, row - 1)):
            if cell != default and self.depth.get(cell) == self.depth[start] - 1:
                candidates.append(cell)
        if len(candidates) == 1:
            return default
        positions = []
        for actor in enemies:
            ex, ey = actor["pos"][:2]
            vx, vy = actor["vel"][:2]
            positions.extend(((ex, ey), (ex + 0.5 * vx, ey + 0.5 * vy)))

        def clearance(cell):
            cx, cy = self.cell_center(cell)
            return min((cx - ex) ** 2 + (cy - ey) ** 2 for ex, ey in positions)

        return max(candidates, key=clearance)

    def direction(self, x, y, dest):
        cx, cy = self.cell_center(dest)
        dx = cx - x
        dy = cy - y
        norm = max(1.0, math.hypot(dx, dy))
        return [dx / norm, dy / norm]
