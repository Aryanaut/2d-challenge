"""Prompts for the AI server and readers for its answers (no ROS).

The reply always uses the same four keys (so the server needs no change). The choice is the number
after "frontier" / "target" at the start of scene_description, e.g. "frontier 3: wide opening away
from walls", or the tile letter for the initial placement, e.g. "tile B: walls on three sides".
"""
import os
import re
from string import Template
from typing import Dict, List, Optional, Sequence, Tuple

PROMPT_FILES = {
    'describe_view': 'environment_report.txt',
    'describe_tile': 'tile_report.txt',
    'describe_scene': 'scene_report.txt',
    'describe_scene_refs': 'scene_reference_report.txt',   # describe_scene with reference photos (collage)
    'place_initial_tile': 'tile_init.txt',
    'choose_tile': 'tile_choice.txt',
    'choose_frontier': 'frontier_choice.txt',
}


def prompts_dir() -> str:
    """The installed share/lite3_llm_tools/prompts, or the source tree's prompts/ when running from source."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('lite3_llm_tools'), 'prompts')
        if os.path.isdir(path):
            return path
    except Exception:  # noqa: BLE001 - not installed (unit tests, running from the source tree)
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prompts')


def prompt_path(task: str, override: str = '') -> str:
    return override or os.path.join(prompts_dir(), PROMPT_FILES[task])

SCENE_TYPES = ('clothes_shop', 'fitness_centre', 'kitchen', 'market', 'workshop', 'clinic', 'cleaning_room', 'furniture_shop')
_SCENE = re.compile(r'^\W*(' + '|'.join(SCENE_TYPES + ('traversable',)) + r')\b', re.IGNORECASE)


def parse_scene_type(scene_description: str, allowed: Optional[Sequence[str]] = None) -> Optional[str]:
    """One of `allowed` (default SCENE_TYPES), 'traversable' (none of them fits), or None if the answer does not
    start with either. A scene type outside `allowed` is also None."""
    if allowed is None:
        m = _SCENE.match(str(scene_description).strip())
        return m.group(1).lower() if m else None
    m = re.match(r'^\W*([A-Za-z][A-Za-z0-9_]*)', str(scene_description).strip())
    word = m.group(1).lower() if m else ''
    return word if word == 'traversable' or word in {a.lower() for a in allowed} else None


def build_scene_reference_prompt(template: str, scene_types: Sequence[str]) -> str:
    return Template(template).safe_substitute(scene_types=', '.join(scene_types),
                                              scene_words=', '.join(list(scene_types) + ['traversable']))


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
    lines = ['id | tile | steps along the Nav2 path from the robot | distance from robot (m) | '
             'distance to nearest wall (m) | edge of unexplored space in tile']
    for t in targets:
        lines.append(f"{t['id']} | {t['tile']} | {t['steps']} | {t['distance_m']:.1f} | "
                     f"{t['clearance_m']:.2f} | {'yes' if t['has_frontier'] else 'no'}")
    return '\n'.join(lines)


def path_hints(targets: List[dict]) -> str:
    """Per target: the Nav2 path's 2-3 evenly spaced points (map frame, metres) the AI may use as waypoints."""
    lines = []
    for t in targets:
        hint = t.get('hint_xy')
        if hint:
            pts = '  '.join(f'({x:.2f}, {y:.2f})' for x, y in hint)
            lines.append(f"{t['id']}: robot ({t['start_xy'][0]:.2f}, {t['start_xy'][1]:.2f}) -> {pts} "
                         f"-> goal ({t['goal_xy'][0]:.2f}, {t['goal_xy'][1]:.2f})")
    return '\n'.join(lines) or 'not available: choose waypoints on a straight line from the robot to the goal'


def build_tile_prompt(template: str, targets: List[dict], environment: str, min_clearance_m: float,
                      stops_so_far: int, tile_summary: str) -> str:
    return Template(template).safe_substitute(
        targets=target_table(targets), paths=path_hints(targets), tiles=tile_summary, environment=environment or 'unknown',
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


def build_initial_tile_prompt(template: str, candidates: List[dict], size_m: float) -> str:
    """Prompt for the initial 1 m tile placement; candidates are {id, kind, sides{N,E,S,W}}."""
    table = '\n'.join(f"{c['id']} | {c['kind']} | " + ', '.join(f'{k}={v}' for k, v in c.get('sides', {}).items())
                      for c in candidates)
    return Template(template).safe_substitute(
        candidates='id | placement | sides measured from the map\n' + table,
        ids=', '.join(str(c['id']) for c in candidates), size=f'{size_m:g}')


_TILE = re.compile(r'\btile\s*([A-D])\b', re.IGNORECASE)
_SIDE = re.compile(r'\b([NESW])\s*[=:]\s*(wall|open|frontier|unknown)', re.IGNORECASE)


def parse_tile_choice(scene_description: str, evidence: Sequence[str], valid_ids) -> Tuple[Optional[str], Dict[str, str]]:
    """('A'..'D' or None, {side: wall/open/frontier/unknown}) from the AI's four-key answer."""
    choice = None
    texts = [str(scene_description)] + [str(e) for e in evidence]
    for text in texts:
        m = _TILE.search(text)
        if m and m.group(1).upper() in valid_ids:
            choice = m.group(1).upper()
            break
    if choice is None:
        first = str(scene_description).strip()[:1].upper()
        if first in valid_ids and (len(str(scene_description).strip()) == 1
                                   or not str(scene_description).strip()[1:2].isalpha()):
            choice = first
    sides = {}
    for text in texts:
        for side, state in _SIDE.findall(text):
            sides.setdefault(side.upper(), state.lower())
    return choice, sides


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
