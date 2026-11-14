"""Build the 'which frontier next?' prompt and read the AI's choice (no ROS).

The reply uses the same four keys as every other prompt (so the server needs no
change). The choice is the number after "frontier" at the start of
scene_description, e.g. "frontier 3: wide opening away from walls".
"""
import re
from string import Template
from typing import List, Optional

_CHOICE = re.compile(r'\b(?:frontier|target)\s*(?:#|no\.?|number)?\s*(\d+)', re.IGNORECASE)


def candidate_table(candidates: List[dict]) -> str:
    with_tiles = any('tile' in c for c in candidates)
    header = 'id | distance from robot (m) | frontier width (m) | distance to nearest wall (m)'
    lines = [header + (' | tile | tile status' if with_tiles else '')]
    for c in candidates:
        row = f"{c['id']} | {c['distance_m']:.1f} | {c['size_m']:.1f} | {c['clearance_m']:.2f}"
        if with_tiles:
            row += f" | {c.get('tile', '?')} | {c.get('tile_status', 'unknown').replace('_', ' ')}"
        lines.append(row)
    return '\n'.join(lines)


def target_table(targets: List[dict]) -> str:
    lines = ['id | tile | steps through connected tiles | distance from robot (m) | '
             'distance to nearest wall (m) | edge of unexplored space in tile']
    for t in targets:
        lines.append(f"{t['id']} | {t['tile']} | {t['steps']} | {t['distance_m']:.1f} | "
                     f"{t['clearance_m']:.2f} | {'yes' if t['has_frontier'] else 'no'}")
    return '\n'.join(lines)


def build_tile_prompt(template: str, targets: List[dict], environment: str, min_clearance_m: float,
                      stops_so_far: int, tile_summary: str) -> str:
    return Template(template).safe_substitute(
        targets=target_table(targets), tiles=tile_summary, environment=environment or 'unknown',
        min_clearance=f'{min_clearance_m:.2f}', stops=str(stops_so_far),
        ids=', '.join(str(t['id']) for t in targets))


def build_prompt(template: str, candidates: List[dict], environment: str, min_clearance_m: float,
                 stops_so_far: int, tile_summary: str = '') -> str:
    return Template(template).safe_substitute(
        tiles=tile_summary or 'tile grid not available',
        candidates=candidate_table(candidates),
        environment=environment or 'unknown',
        min_clearance=f'{min_clearance_m:.2f}',
        stops=str(stops_so_far),
        ids=', '.join(str(c['id']) for c in candidates))


def parse_choice(scene_description: str, evidence: List[str], valid_ids) -> Optional[int]:
    """Return the chosen frontier id, or None for 'none' / an id that isn't offered."""
    valid = set(valid_ids)
    for text in [scene_description] + list(evidence):
        match = _CHOICE.search(str(text))
        if match:
            choice = int(match.group(1))
            return choice if choice in valid else None
    text = str(scene_description).strip()
    if text.isdigit() and int(text) in valid:
        return int(text)
    return None
