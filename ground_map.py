"""Ground routing on a conservative four-metre occupancy grid."""
from collections import deque
import json
import math
from pathlib import Path

STEP = 4


class GroundMap:
    def __init__(self, world, radius):
        bounds = world["bounds"]
        self.x0, self.y0 = bounds["minX"], bounds["minY"]
        self.width = math.ceil((bounds["maxX"] - self.x0) / STEP)
        self.height = math.ceil((bounds["maxY"] - self.y0) / STEP)
        self.blocked = set()
        for prism in world["prisms"]:
            if prism["kind"] == "bridge":
                continue
            solid = self.fill(prism["ring"])
            for hole in prism["holes"]:
                solid.difference_update(self.fill(hole))
            self.blocked.update(solid)
        # Include adjacent cells for the mech body and discretization error.
        padding = math.ceil(radius / STEP)
        occupied = self.solid = self.blocked
        self.blocked = {(x + dx, y + dy) for x, y in occupied
                        for dx in range(-padding, padding + 1)
                        for dy in range(-padding, padding + 1)}
        self.goal = None
        self.frontier = deque()
        self.next_cell = {}

    @classmethod
    def load(cls, hello):
        path = Path(hello["map"]["dir"]) / "v2" / "world.json"
        with path.open(encoding="utf-8") as stream:
            return cls(json.load(stream), hello["config"]["mech"]["radius"])

    def cell(self, x, y):
        return math.floor((x-self.x0)/STEP), math.floor((y-self.y0)/STEP)

    def fill(self, ring):
        vertices = list(zip(ring[::2], ring[1::2]))
        edges = list(zip(vertices, vertices[1:] + vertices[:1]))
        rows = [self.cell(x,y)[1] for x,y in vertices]
        cells = set()
        for row in range(min(rows), max(rows)+1):
            y = self.y0 + (row + 0.5)*STEP
            crosses = sorted(ax+(y-ay)*(bx-ax)/(by-ay) for (ax,ay),(bx,by) in edges
                             if (ay > y) != (by > y))
            for left,right in zip(crosses[::2], crosses[1::2]):
                for col in range(self.cell(left,y)[0], self.cell(right,y)[0]+1):
                    cells.add((col,row))
        return cells

    def open(self, cell):
        x,y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.blocked

    def nearest_open(self, cell):
        x,y = cell
        for r in range(1, 32):
            edge = [(x+d,y+s*r) for d in range(-r,r+1) for s in (-1,1)]
            edge.extend((x+s*r,y+d) for d in range(1-r,r) for s in (-1,1))
            candidates = [p for p in edge if self.open(p)]
            if candidates:
                return min(candidates,key=lambda p: (p[0]-x)**2+(p[1]-y)**2)
        return None

    def steer(self, obs, fallback):
        x,y,_ = obs["self"]["pos"]
        start = self.cell(x,y)
        goal = self.cell(*obs["zone"]["next_center"])
        if self.goal != goal:
            self.goal = goal
            dest = goal if self.open(goal) else self.nearest_open(goal)
            self.next_cell.clear()
            self.frontier.clear()
            if dest is not None:
                self.frontier.append(dest)
                self.next_cell[dest] = dest
        gx, gy = obs["zone"]["next_center"]
        distance = math.hypot(gx-x,gy-y)
        if distance < 16:
            steps = max(1,math.ceil(distance))
            if all(self.cell(x+(gx-x)*i/steps,y+(gy-y)*i/steps) not in self.solid
                   for i in range(steps+1)):
                return fallback
        if start in self.solid:
            return fallback
        if not self.open(start):
            exit_cell = self.nearest_open(start)
            if exit_cell is not None:
                return self.direction(x,y,exit_cell)
            return fallback
        # Incrementally build a shortest-path field shared by all decisions for this goal.
        for _ in range(1400):
            if start in self.next_cell or not self.frontier:
                break
            a,b = here = self.frontier.popleft()
            for neighbor in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                if self.open(neighbor) and neighbor not in self.next_cell:
                    self.next_cell[neighbor] = here
                    self.frontier.append(neighbor)
        dest = self.next_cell.get(start)
        if dest is None or (dest == start and start == goal):
            return fallback
        return self.direction(x,y,dest)

    def direction(self,x,y,dest):
        dx = self.x0+(dest[0]+0.5)*STEP-x
        dy = self.y0+(dest[1]+0.5)*STEP-y
        norm = max(1.0, math.hypot(dx,dy))
        return [dx/norm,dy/norm]
