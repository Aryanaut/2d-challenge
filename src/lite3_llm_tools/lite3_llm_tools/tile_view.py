"""A read-only view of a tile grid, rebuilt from the JSON the explorer sends (no ROS).

The explorer owns the tile logic. The AI tools only need the geometry to draw it: the tile corners, centres
and edges in the map frame. Wire format (same as the explorer's /tile_map "grid" object):
    {"origin_xy": [x, y], "yaw_deg": 4.0, "size_m": 1.0}
Tiles are (i, j) integer pairs; tile (0, 0) is the initial tile, i along the grid's +x, j along its +y.
"""
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

Tile = Tuple[int, int]


@dataclass
class TileView:
    origin_x: float
    origin_y: float
    yaw: float
    size: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> 'TileView':
        ox, oy = data['origin_xy']
        return cls(float(ox), float(oy), math.radians(float(data.get('yaw_deg', 0.0))),
                   float(data.get('size_m', 1.0)))

    def to_world(self, u: float, v: float) -> Tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.origin_x + c * u - s * v, self.origin_y + s * u + c * v

    def center(self, tile: Tile) -> Tuple[float, float]:
        return self.to_world((tile[0] + 0.5) * self.size, (tile[1] + 0.5) * self.size)

    def corners(self, tile: Tile) -> List[Tuple[float, float]]:
        i, j = tile
        return [self.to_world(u * self.size, v * self.size)
                for u, v in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]

    def edge(self, tile: Tile, side: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        sw, se, ne, nw = self.corners(tile)
        return {'S': (sw, se), 'E': (se, ne), 'N': (nw, ne), 'W': (sw, nw)}[side]


def parse_tile_label(label) -> Optional[Tile]:
    """'1,0' or [1, 0] -> (1, 0)."""
    try:
        if isinstance(label, str):
            i, j = label.split(',')
            return int(i), int(j)
        i, j = label
        return int(i), int(j)
    except (TypeError, ValueError):
        return None
