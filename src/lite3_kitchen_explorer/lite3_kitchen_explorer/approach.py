"""Approach route to the next tile: 2-3 waypoints along the Nav2 path, each with a yaw, plus a final pose (no ROS).

Yaw convention everywhere: map frame, degrees in the AI's JSON (0 = map +x, counter-clockwise, 90 = map +y,
i.e. "up" in the rendered map), radians in the returned poses.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

Pose2 = Tuple[float, float, float]            # x, y, yaw (rad)
MIN_WAYPOINTS = 2
MAX_WAYPOINTS = 3
WAYPOINT_SPACING_M = 0.5      # about one intermediate waypoint per this much path; the robot drives through each


def waypoint_count(points: Sequence[Sequence[float]]) -> int:
    """2 or 3 intermediate waypoints: about one per WAYPOINT_SPACING_M of path (a 1 m tile hop gets 2)."""
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    return max(MIN_WAYPOINTS, min(MAX_WAYPOINTS, round(length / WAYPOINT_SPACING_M)))


def _cumulative(points: Sequence[Sequence[float]]) -> List[float]:
    acc = [0.0]
    for a, b in zip(points, points[1:]):
        acc.append(acc[-1] + math.dist(a, b))
    return acc


def sample_path(points: Sequence[Sequence[float]], n: Optional[int] = None) -> List[Tuple[float, float, int]]:
    """n points at equal path length fractions 1/(n+1) .. n/(n+1), each with its index into `points`."""
    if len(points) < 2:
        return []
    n = waypoint_count(points) if n is None else n
    acc = _cumulative(points)
    total = acc[-1]
    out, k = [], 0
    for i in range(1, n + 1):
        d = total * i / (n + 1)
        while k < len(points) - 2 and acc[k + 1] < d:
            k += 1
        seg = acc[k + 1] - acc[k]
        t = 0.0 if seg < 1e-9 else (d - acc[k]) / seg
        x = points[k][0] + t * (points[k + 1][0] - points[k][0])
        y = points[k][1] + t * (points[k + 1][1] - points[k][1])
        out.append((float(x), float(y), k + (1 if t > 0.5 else 0)))
    return out


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def snap_to_axis(yaw: float, grid_yaw: float) -> float:
    """The tile-grid axis direction (grid_yaw + k*90 deg) closest to yaw."""
    k = round((wrap(yaw - grid_yaw)) / (math.pi / 2))
    return wrap(grid_yaw + k * math.pi / 2)


def default_route(path: Sequence[Sequence[float]], goal_xy: Sequence[float], grid_yaw: float,
                  n: Optional[int] = None) -> List[Pose2]:
    """2-3 (waypoint_count) evenly spaced waypoints facing along the path, then the goal facing along the nearest grid axis."""
    pts = [tuple(p[:2]) for p in path] or []
    goal = (float(goal_xy[0]), float(goal_xy[1]))
    if not pts or math.dist(pts[-1], goal) > 1e-6:
        pts.append(goal)
    if len(pts) < 2:
        return [(goal[0], goal[1], 0.0)]
    samples = sample_path(pts, n)
    stops = [(x, y) for x, y, _ in samples] + [goal]
    route = []
    for i, (x, y) in enumerate(stops):
        nxt = stops[i + 1] if i + 1 < len(stops) else None
        if nxt is not None:
            yaw = math.atan2(nxt[1] - y, nxt[0] - x)
        else:
            back = stops[i - 1] if i else pts[0]
            yaw = snap_to_axis(math.atan2(y - back[1], x - back[0]), grid_yaw)
        route.append((x, y, yaw))
    return route


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_route(extras: dict, path: Sequence[Sequence[float]], goal_xy: Sequence[float], grid_yaw: float,
                n: Optional[int] = None, snap_m: float = 0.5) -> Tuple[List[Pose2], str]:
    """Route from the AI's "target_yaw_deg" and "waypoints" [{x, y, yaw_deg}, ...].

    Each AI waypoint is snapped onto the Nav2 path (so it is always reachable and clear of walls) and
    must lie within snap_m of it and come after the previous one; otherwise the evenly spaced default
    position is used for that waypoint. A missing or invalid yaw also falls back to the default yaw.
    Returns (route, note) where note says what was taken from the AI.
    """
    base = default_route(path, goal_xy, grid_yaw, n)
    if len(base) < 2:
        return base, 'default (path too short)'
    n = len(base) - 1
    pts = [tuple(p[:2]) for p in path]
    wps = (extras or {}).get('waypoints')
    wps = wps if isinstance(wps, list) else []
    route = list(base)
    used = 0
    last_idx = -1
    for i in range(n):
        item = wps[i] if i < len(wps) and isinstance(wps[i], dict) else {}
        x, y, yaw_d = _num(item.get('x')), _num(item.get('y')), _num(item.get('yaw_deg'))
        bx, by, byaw = base[i]
        if x is not None and y is not None and pts:
            idx = min(range(len(pts)), key=lambda k: math.dist(pts[k], (x, y)))
            if math.dist(pts[idx], (x, y)) <= snap_m and idx > last_idx:
                bx, by = pts[idx]
                last_idx = idx
                used += 1
        if yaw_d is not None:
            byaw = wrap(math.radians(yaw_d))
            used += 1
        route[i] = (bx, by, byaw)
    gx, gy, gyaw = base[n]
    t = _num((extras or {}).get('target_yaw_deg'))
    if t is not None:
        gyaw = wrap(math.radians(t))
        used += 1
    route[n] = (gx, gy, gyaw)
    return route, f'{used}/{2 * n + 1} values from the AI' if used else 'default (no usable AI route)'


def route_json(route: Sequence[Pose2]) -> List[dict]:
    return [{'x': round(x, 3), 'y': round(y, 3), 'yaw_deg': round(math.degrees(yaw), 1)} for x, y, yaw in route]


def filter_route(route: Sequence[Pose2], start_xy: Sequence[float], min_sep_m: float = 0.15,
                 max_heading_err_deg: float = 90.0) -> Tuple[List[Pose2], List[str]]:
    """Drop intermediate waypoints that would make the robot turn away from the tile (the last pose is kept).

    A waypoint (x, y, yaw) is dropped when
      * it is closer than min_sep_m to the previous kept pose (or the start): its yaw is then noise, and
      * its heading points away from the goal: with b = atan2(gy - y, gx - x) the bearing to the goal,
        cos(yaw - b) < cos(max_heading_err_deg), i.e. the unit heading vector has a negative component
        towards the goal (default: more than 90 deg off).
    Returns (kept route, one human-readable reason per dropped waypoint).
    """
    if len(route) < 2:
        return list(route), []
    gx, gy, _ = route[-1]
    limit = math.cos(math.radians(max_heading_err_deg))
    prev = (float(start_xy[0]), float(start_xy[1]))
    kept: List[Pose2] = []
    dropped: List[str] = []
    for i, (x, y, yaw) in enumerate(route[:-1], 1):
        if math.dist(prev, (x, y)) < min_sep_m:
            dropped.append(f'#{i} {math.dist(prev, (x, y)):.2f} m from previous')
            continue
        if math.dist((x, y), (gx, gy)) > 1e-6:
            bearing = math.atan2(gy - y, gx - x)
            if math.cos(yaw - bearing) < limit:
                err = math.degrees(abs(wrap(yaw - bearing)))
                dropped.append(f'#{i} yaw {math.degrees(yaw):+.0f} deg is {err:.0f} deg off the goal bearing')
                continue
        kept.append((x, y, yaw))
        prev = (x, y)
    kept.append(route[-1])
    return kept, dropped
