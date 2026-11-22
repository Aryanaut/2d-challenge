"""Maze definition for lite3_sim (no ROS): a grid of square tiles with wall panels along the tile edges.

A maze file is YAML. The layout is ASCII art, one tile = 4 characters wide and 2 lines tall:

    +---+---+---+---+          wall rows:  '+' corner posts, between them (3 chars, one tile edge):
    | S         ~   |                         '---' wall panel     '~~~' curtain     '===' slit panel
    +   +---+   +   +                         '   ' open
    |   |R1 |       |          cell rows:  between tiles (1 char, one tile edge):
    +   +   +---+   +                         '|'   wall panel     '~'   curtain     '!'   slit panel
    |   0   | # |   |                         ' '   open
    +---+---+---+---+                      inside a tile (3 chars):
                                              'S' start (facing start.heading), or '>' '<' '^' 'v' start facing E W N S
                                              '#' solid (not floor: boxed in by walls)
                                              '0' scene tile (clutter on the floor: looked at, never entered)
                                              anything else: a label drawn in RViz (e.g. R1, ST for stairs)

The top line is north (+y), the left column west. Tile (i, j): i counts east from the west column, j north from
the south row, both from 0. The maze frame has its origin at the south-west corner of tile (0, 0).

  wall panel   blocks the lidar and the robot
  curtain      blocks the lidar (the 2D scan sees a wall) but the robot walks through it; the simulated depth
               camera reports it to the mission manager's curtain mode on /curtain/points
  slit panel   the aluminium frame panels: blocks the robot, but leaves a `slit_m` gap at both ends that the lidar
               sees through
  scene tile   '0': clutter on the tile (a table, boxes, ...). The simulated lidar maps it cell by cell as a fixed
               pattern of occupied and free cells (world.py, `scene` settings), so the mission manager sees a scanned,
               mostly non-traversable tile = a scene tile, and its scene mode looks at it from a neighbouring tile.
               The robot collides with the cluttered part (the tile minus a margin along its edges).

Everything else is optional:

    name: my_maze
    tile_size: 1.02            # m (the real tiles)
    wall_thickness: 0.02       # m
    slit_m: 0.04               # gap at each end of a slit panel
    layout: |
      ...
    start:                     # overrides / completes the S in the layout
      tile: [0, 0]             # (i, j)
      heading: E               # N E S W, or degrees (0 = east, counter-clockwise)
      offset_m: [0.0, 0.0]     # from the tile centre, in the maze frame
      yaw_offset_deg: 0.0      # added to the heading: start not square to the walls
    extra:                     # free-form geometry in metres, maze frame
      walls:    [[x0, y0, x1, y1], ...]
      curtains: [[x0, y0, x1, y1], ...]
      boxes:    [[x, y, w, h], ...]        # solid obstacles: x, y = south-west corner
"""
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

WALL, CURTAIN, SLIT = 'wall', 'curtain', 'slit'
HEADINGS = {'E': 0.0, 'N': 90.0, 'W': 180.0, 'S': -90.0}
ARROWS = {'>': 'E', '<': 'W', '^': 'N', 'v': 'S', 'V': 'S'}
H_CHARS = {'-': WALL, '~': CURTAIN, '=': SLIT}
V_CHARS = {'|': WALL, '~': CURTAIN, '!': SLIT, ':': SLIT}

Tile = Tuple[int, int]


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = WALL

    def as_tuple(self):
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Maze:
    name: str
    tile_size: float
    cols: int
    rows: int
    segments: List[Segment]
    start: Tuple[float, float, float]                       # x, y, yaw (rad) in the maze frame
    edges: Dict[Tuple[int, int, str], str] = field(default_factory=dict)   # (i, j, side) -> kind, every tile edge
    labels: Dict[Tile, str] = field(default_factory=dict)
    solid: set = field(default_factory=set)
    scenes: set = field(default_factory=set)                 # scene tiles ('0'): clutter, never entered
    wall_thickness: float = 0.02
    slit_m: float = 0.04

    @property
    def size_m(self) -> Tuple[float, float]:
        return self.cols * self.tile_size, self.rows * self.tile_size

    def tile_centre(self, tile: Tile) -> Tuple[float, float]:
        return (tile[0] + 0.5) * self.tile_size, (tile[1] + 0.5) * self.tile_size

    def edge(self, tile: Tile, side: str) -> Optional[str]:
        """Kind of the edge on `side` (N E S W) of `tile`: wall / curtain / slit, or None when open."""
        return self.edges.get((tile[0], tile[1], side))

    def floor_tiles(self) -> List[Tile]:
        return [(i, j) for i in range(self.cols) for j in range(self.rows) if (i, j) not in self.solid]

    def reachable_tiles(self, through_curtains: bool = True) -> set:
        """Floor tiles connected to the start tile through open edges (and curtains)."""
        start = (int(self.start[0] // self.tile_size), int(self.start[1] // self.tile_size))
        passable = {None, CURTAIN} if through_curtains else {None}
        seen, todo = {start}, [start]
        steps = {'E': (1, 0), 'W': (-1, 0), 'N': (0, 1), 'S': (0, -1)}
        while todo:
            t = todo.pop()
            for side, (di, dj) in steps.items():
                n = (t[0] + di, t[1] + dj)
                if (0 <= n[0] < self.cols and 0 <= n[1] < self.rows and n not in seen and n not in self.solid
                        and n not in self.scenes and self.edge(t, side) in passable):
                    seen.add(n)
                    todo.append(n)
        return seen

    def summary(self) -> str:
        kinds = {}
        for s in self.segments:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        w, h = self.size_m
        reach = self.reachable_tiles()
        return (f'maze {self.name!r}: {self.cols} x {self.rows} tiles of {self.tile_size:g} m ({w:.2f} x {h:.2f} m), '
                f'{len(self.floor_tiles())} floor tiles ({len(reach)} reachable from the start, '
                f'{len(self.reachable_tiles(False))} without crossing a curtain'
                + (f', {len(self.scenes)} scene tile(s)' if self.scenes else '') + '), segments: '
                + ', '.join(f'{n} {k}' for k, n in sorted(kinds.items()))
                + f'; start ({self.start[0]:.2f}, {self.start[1]:.2f}) heading {math.degrees(self.start[2]):.0f} deg')


class MazeError(ValueError):
    pass


def _heading_deg(value) -> float:
    if isinstance(value, str):
        key = value.strip().upper()
        if key in HEADINGS:
            return HEADINGS[key]
        try:
            return float(key)
        except ValueError:
            raise MazeError(f'start.heading must be N/E/S/W or degrees, not {value!r}') from None
    return float(value)


def parse_layout(text: str):
    """ASCII layout -> (cols, rows, edges {(i, j, side): kind}, labels, solid, start (tile, heading or None),
    scenes)."""
    lines = textwrap.dedent(text.expandtabs(4)).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 3 or len(lines) % 2 == 0:
        raise MazeError(f'the layout needs an odd number of lines (wall row, cell row, ..., wall row); got {len(lines)}')
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width + 4) for ln in lines]
    cols = (width - 1) // 4
    rows = (len(lines) - 1) // 2
    if cols < 1:
        raise MazeError('the layout is narrower than one tile ("+---+")')
    edges: Dict[Tuple[int, int, str], str] = {}
    labels: Dict[Tile, str] = {}
    solid, scenes, start = set(), set(), None

    def set_edge(i, j, side, kind):
        edges[(i, j, side)] = kind
        other = {'N': (i, j + 1, 'S'), 'S': (i, j - 1, 'N'), 'E': (i + 1, j, 'W'), 'W': (i - 1, j, 'E')}[side]
        edges[other] = kind

    for k, line in enumerate(lines):
        if k % 2 == 0:                                  # wall row: the edge between tile rows
            j_north = rows - 1 - k // 2 + 1             # the tile row north of this line (j), -> its S side
            for i in range(cols):
                chunk = line[4 * i + 1:4 * i + 4]
                kind = next((H_CHARS[ch] for ch in chunk if ch in H_CHARS), None)
                if kind is None and chunk.strip():
                    raise MazeError(f'line {k + 1}, tile column {i}: {chunk!r} is not a wall ("---"), curtain ("~~~"), '
                                    f'slit panel ("===") or open ("   ")')
                if kind:
                    set_edge(i, j_north, 'S', kind)
        else:                                           # cell row
            j = rows - 1 - k // 2
            for i in range(cols + 1):
                ch = line[4 * i]
                if ch in V_CHARS:
                    set_edge(i, j, 'W', V_CHARS[ch])
                elif ch.strip():
                    raise MazeError(f'line {k + 1}, column {4 * i + 1}: {ch!r} is not a wall "|", curtain "~", '
                                    f'slit panel "!" or open " "')
            for i in range(cols):
                cell = line[4 * i + 1:4 * i + 4].strip()
                if not cell:
                    continue
                if cell == '#':
                    solid.add((i, j))
                elif cell == '0':
                    scenes.add((i, j))
                elif cell == 'S' or cell in ARROWS:
                    if start is not None:
                        raise MazeError(f'two start tiles in the layout: {start[0]} and {(i, j)}')
                    start = ((i, j), ARROWS.get(cell))
                else:
                    labels[(i, j)] = cell
    # keep only edges that belong to tiles inside the maze
    edges = {key: v for key, v in edges.items() if 0 <= key[0] < cols and 0 <= key[1] < rows}
    for t in solid:                                     # a solid cell is boxed in
        for side in 'NESW':
            edges.setdefault((t[0], t[1], side), WALL)
            n = {'N': (t[0], t[1] + 1, 'S'), 'S': (t[0], t[1] - 1, 'N'),
                 'E': (t[0] + 1, t[1], 'W'), 'W': (t[0] - 1, t[1], 'E')}[side]
            if 0 <= n[0] < cols and 0 <= n[1] < rows:
                edges.setdefault(n, WALL)
    return cols, rows, edges, labels, solid, start, scenes


def _edge_segments(cols, rows, edges, size, slit_m) -> List[Segment]:
    """One segment per tile edge (each shared edge once)."""
    out = []
    for i in range(cols + 1):
        for j in range(rows + 1):
            # south edge of (i, j) = the horizontal line y = j * size, x in [i, i+1] * size
            if i < cols:
                kind = edges.get((i, j, 'S')) if j < rows else edges.get((i, j - 1, 'N'))
                if kind:
                    out.append(_panel(i * size, j * size, (i + 1) * size, j * size, kind, slit_m))
            # west edge of (i, j) = the vertical line x = i * size
            if j < rows:
                kind = edges.get((i, j, 'W')) if i < cols else edges.get((i - 1, j, 'E'))
                if kind:
                    out.append(_panel(i * size, j * size, i * size, (j + 1) * size, kind, slit_m))
    return out


def _panel(x0, y0, x1, y1, kind, slit_m) -> Segment:
    if kind == SLIT and slit_m > 0:
        length = math.hypot(x1 - x0, y1 - y0)
        f = min(0.45, slit_m / max(length, 1e-9))
        dx, dy = (x1 - x0) * f, (y1 - y0) * f
        return Segment(x0 + dx, y0 + dy, x1 - dx, y1 - dy, SLIT)
    return Segment(x0, y0, x1, y1, kind)


def load_maze_dict(data: dict, name: str = 'maze') -> Maze:
    if not isinstance(data, dict) or 'layout' not in data:
        raise MazeError('a maze file needs a "layout:" block (ASCII art)')
    size = float(data.get('tile_size', 1.02))
    slit_m = float(data.get('slit_m', 0.04))
    cols, rows, edges, labels, solid, start, scenes = parse_layout(str(data['layout']))
    segments = _edge_segments(cols, rows, edges, size, slit_m)

    extra = data.get('extra') or {}
    for key, kind in (('walls', WALL), ('curtains', CURTAIN), ('slits', SLIT)):
        for seg in extra.get(key) or []:
            if len(seg) != 4:
                raise MazeError(f'extra.{key}: each entry is [x0, y0, x1, y1], got {seg!r}')
            segments.append(_panel(*(float(v) for v in seg), kind, slit_m) if kind == SLIT
                            else Segment(*(float(v) for v in seg), kind))
    for box in extra.get('boxes') or []:
        if len(box) != 4:
            raise MazeError(f'extra.boxes: each entry is [x, y, w, h], got {box!r}')
        x, y, w, h = (float(v) for v in box)
        segments += [Segment(x, y, x + w, y), Segment(x + w, y, x + w, y + h),
                     Segment(x + w, y + h, x, y + h), Segment(x, y + h, x, y)]

    sc = data.get('start') or {}
    tile = tuple(int(v) for v in sc['tile']) if 'tile' in sc else (start[0] if start else None)
    if tile is None:
        raise MazeError('no start: put S (or > < ^ v) in a tile of the layout, or set start.tile: [i, j]')
    if not (0 <= tile[0] < cols and 0 <= tile[1] < rows) or tile in solid:
        raise MazeError(f'start tile {tile} is not a floor tile of the {cols} x {rows} maze')
    if tile in scenes:
        raise MazeError(f'start tile {tile} is a scene tile ("0"): the robot never stands on one')
    heading = sc.get('heading', (start[1] if start and start[1] else None) or 'E')
    yaw = math.radians(_heading_deg(heading) + float(sc.get('yaw_offset_deg', 0.0)))
    off = [float(v) for v in sc.get('offset_m', [0.0, 0.0])]
    x, y = (tile[0] + 0.5) * size + off[0], (tile[1] + 0.5) * size + off[1]
    return Maze(name=str(data.get('name') or name), tile_size=size, cols=cols, rows=rows, segments=segments,
                start=(x, y, yaw), edges=edges, labels=labels, solid=solid, scenes=scenes,
                wall_thickness=float(data.get('wall_thickness', 0.02)), slit_m=slit_m)


def load_maze(path: str) -> Maze:
    import os
    with open(os.path.expanduser(path), encoding='utf-8') as fh:
        data = yaml.safe_load(fh)
    return load_maze_dict(data, name=os.path.splitext(os.path.basename(path))[0])
