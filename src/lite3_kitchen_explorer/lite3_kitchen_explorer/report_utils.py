"""Turn the AI answers from one stop into a stop report, and render the run report (no ROS)."""
from collections import Counter
import math
from typing import List, Optional


def normalise_scene(text: str) -> str:
    return ' '.join(str(text).strip().lower().rstrip('.').split())


def summarize_views(views: List[dict]) -> dict:
    """Pick the environment type for one stop from its look-around views.

    Each view is {heading_deg, scene_description, relevant_to_mission, evidence, confidence, ...}
    or {heading_deg, error}. relevant_to_mission is true when the view clearly shows what kind
    of environment it is (see the prompt), so those views decide; the most confident wins and
    its description becomes the stop's environment.
    """
    answered = [v for v in views if 'error' not in v]
    identified = [v for v in answered
                  if v.get('relevant_to_mission') and normalise_scene(v['scene_description']) != 'unknown']
    pool = identified or [v for v in answered if normalise_scene(v['scene_description']) != 'unknown']
    if not pool:
        return {'environment': 'unknown', 'confidence': 0.0, 'evidence': [], 'identified_views': 0,
                'answered_views': len(answered), 'total_views': len(views), 'all_descriptions': {}}
    best = max(pool, key=lambda v: v['confidence'])
    counts = Counter(normalise_scene(v['scene_description']) for v in answered
                     if normalise_scene(v['scene_description']) != 'unknown')
    return {
        'environment': best['scene_description'],
        'confidence': round(float(best['confidence']), 3),
        'evidence': list(best.get('evidence', []))[:3],
        'best_heading_deg': best.get('heading_deg'),
        'identified_views': len(identified),
        'answered_views': len(answered),
        'total_views': len(views),
        'all_descriptions': dict(counts.most_common()),
    }


def _fmt_pose(pose: Optional[list]) -> str:
    if not pose:
        return 'unknown'
    return f'({pose[0]:.2f}, {pose[1]:.2f}) facing {math.degrees(pose[2]):.0f}°'


def describe_next(next_frontier: Optional[dict]) -> str:
    if not next_frontier:
        return 'explorer (cost ranking)'
    if next_frontier.get('kind') == 'tile':
        return (f"tile {next_frontier['tile']} ({next_frontier['steps']} step(s), "
                f"{next_frontier['clearance_m']:.2f} m from walls), chosen by {next_frontier['chosen_by']} — "
                f"{next_frontier.get('reason', '')}")
    if next_frontier.get('chosen_by') == 'AI':
        return (f"AI: frontier {next_frontier['id']} ({next_frontier['clearance_m']:.2f} m from walls) — "
                f"{next_frontier.get('reason', '')}")
    return f"explorer — {next_frontier.get('note', '')}"


def render_markdown(reports: List[dict], frame: str = 'map', finished: bool = False) -> str:
    lines = ['# Environment survey', '',
             f'{len(reports)} stop(s) so far, positions in the `{frame}` frame.'
             + (' Exploration complete: every frontier visited.' if finished else ''), '',
             '| Stop | Reason | Position | Environment | Confidence | Evidence | Views identified | Next frontier |',
             '|---|---|---|---|---|---|---|---|']
    for r in reports:
        s = r['summary']
        evidence = '; '.join(s.get('evidence', [])).replace('|', '/')
        lines.append(f"| {r['stop']} | {r['reason']} | {_fmt_pose(r.get('pose'))} | "
                     f"{s['environment'].replace('|', '/')} | {s['confidence']:.2f} | {evidence} | "
                     f"{s['identified_views']}/{s['total_views']} | "
                     f"{describe_next(r.get('next_frontier')).replace('|', '/')} |")
    lines.append('')
    for r in reports:
        lines.append(f"## Stop {r['stop']} ({r['reason']}, {r['wall_time']})")
        lines.append(f"- Next frontier: {describe_next(r.get('next_frontier'))}")
        if r.get('tile'):
            sides = ', '.join(f'{k}={v}' for k, v in (r.get('tile_sides') or {}).items())
            c = r.get('tile_counts', {})
            lines.append(f"- Tile {r['tile']} ({sides}); tiles visited {c.get('visited', 0)}, "
                         f"to visit {c.get('to_visit', 0)}")
        for v in r['views']:
            if 'error' in v:
                lines.append(f"- {v['heading_deg']:.0f}°: query failed ({v['error'][:120]})")
            else:
                lines.append(f"- {v['heading_deg']:.0f}°: {v['scene_description']} "
                             f"(identified={str(v['relevant_to_mission']).lower()}, conf {v['confidence']:.2f}) "
                             f"— {'; '.join(v['evidence'])} [`{v.get('image', '')}`]")
        lines.append('')
    return '\n'.join(lines)
