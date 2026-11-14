"""The tile report: what each tile looks like, compiled into one markdown file (no ROS).

A tile gets an entry the moment the robot reaches its centre. The AI description arrives later, in the
background, and fills the entry in. The whole book is rewritten to one file (default ~/2d/reports/) after every
change, so the file is always current and a crash loses nothing.

Entry status: pending (waiting for the AI), done, error (the AI query failed), no_ai (the AI is off or not running).
"""
import datetime as dt
import math
import os
from typing import Dict, List, Optional

SOURCES = {'start': 'start tile', 'centre reached': 'centre reached', 'goal reached': 'goal reached',
           'manual': 'manual request'}


def _md(text) -> str:
    """One table cell: no pipes or line breaks."""
    return str(text).replace('|', '/').replace('\n', ' ').strip()


def _fmt_pose(pose) -> str:
    return f'({pose[0]:.2f}, {pose[1]:.2f})' if pose else '-'


def _fmt_sides(sides: Optional[dict]) -> str:
    return ' '.join(f'{k}={v}' for k, v in (sides or {}).items()) or '-'


class TileReportBook:
    def __init__(self, path: str, run_dir: str, ai_on: bool, reach_radius_m: float = 0.30):
        self.path = os.path.expanduser(path)
        self.run_dir = run_dir
        self.ai_on = ai_on
        self.reach_radius_m = reach_radius_m
        self.started = dt.datetime.now().isoformat(timespec='seconds')
        self.entries: Dict[str, dict] = {}

    # ------------------------------------------------------------------ updates
    def reached(self, label: str, pose, source: str, sides: Optional[dict] = None) -> dict:
        """Open (or reopen, for a manual re-report) the entry of a tile the robot has just reached."""
        old = self.entries.get(label)
        entry = {'tile': label, 'order': old['order'] if old else len(self.entries) + 1,
                 'reached_at': dt.datetime.now().isoformat(timespec='seconds'), 'pose': list(pose) if pose else None,
                 'source': source, 'sides': dict(sides or {}), 'status': 'pending'}
        self.entries[label] = entry
        return entry

    def set_result(self, label: str, view: dict):
        e = self.entries[label]
        e.update(status='done', description=str(view.get('scene_description', '')),
                 relevant=bool(view.get('relevant_to_mission')), evidence=[str(x) for x in view.get('evidence', [])],
                 confidence=float(view.get('confidence', 0.0)), image=view.get('image', ''),
                 elapsed_s=view.get('elapsed_s'))

    def set_error(self, label: str, message: str):
        self.entries[label].update(status='error', error=str(message))

    def set_no_ai(self, label: str, why: str):
        self.entries[label].update(status='no_ai', error=why)

    def counts(self) -> Dict[str, int]:
        out = {'done': 0, 'pending': 0, 'error': 0, 'no_ai': 0}
        for e in self.entries.values():
            out[e['status']] += 1
        return out

    # ------------------------------------------------------------------ output
    def _description(self, e: dict) -> str:
        return {'done': lambda: e.get('description', ''), 'pending': lambda: '(waiting for the AI ...)',
                'error': lambda: f'(AI query failed: {e.get("error", "")[:120]})',
                'no_ai': lambda: f'({e.get("error", "AI off")})'}[e['status']]()

    def render(self, grid: Optional[dict] = None, tile_counts: Optional[dict] = None,
               to_visit: Optional[List[str]] = None, finished: bool = False) -> str:
        c = self.counts()
        lines = ['# Tile report', '',
                 f'Run `{os.path.basename(self.run_dir.rstrip("/"))}`, started {self.started}, '
                 f'updated {dt.datetime.now().isoformat(timespec="seconds")}'
                 + (' - **exploration complete**.' if finished else '.'), '',
                 f'- **AI descriptions:** {"on" if self.ai_on else "off"} - {c["done"]} done, {c["pending"]} pending, '
                 f'{c["error"]} failed' + (f', {c["no_ai"]} without AI' if c['no_ai'] else ''),
                 f'- **A tile is reached** when the robot is within {self.reach_radius_m:g} m of its centre '
                 '(or Nav2 reports its goal reached there); the initial tile counts from the start.']
        if grid:
            lines.append(f'- **Tile grid:** {grid.get("size_m", 1.0):g} m tiles, origin '
                         f'({grid["origin_xy"][0]:.2f}, {grid["origin_xy"][1]:.2f}) in the map frame, '
                         f'yaw {grid.get("yaw_deg", 0.0):g} deg. Tile "i,j": i along the grid\'s +x, j along its +y.')
        if tile_counts:
            lines.append(f'- **Tiles:** {tile_counts.get("visited", 0)} visited, {tile_counts.get("to_visit", 0)} to visit, '
                         f'{tile_counts.get("blocked", 0)} blocked, {tile_counts.get("unknown", 0)} unknown.')
        lines += ['', '| # | Tile | Reached | Position | Map sides | What it looks like | Conf. |', '|---|---|---|---|---|---|---|']
        done = [e for e in self.entries.values() if e['status'] in ('done', 'no_ai')]   # skip tiles still waiting on, or failed by, the AI
        for e in done:
            conf = f'{e["confidence"]:.2f}' if e['status'] == 'done' else '-'
            lines.append(f'| {e["order"]} | {e["tile"]} | {e["reached_at"][11:]} | {_fmt_pose(e["pose"])} | '
                         f'{_md(_fmt_sides(e["sides"]))} | {_md(self._description(e))} | {conf} |')
        lines.append('')
        for e in done:
            lines.append(f'## {e["order"]}. Tile {e["tile"]}' + (' (start)' if e['source'] == 'start' else ''))
            lines.append(f'- Reached {e["reached_at"]} at {_fmt_pose(e["pose"])} ({SOURCES.get(e["source"], e["source"])})')
            lines.append(f'- Map: {_fmt_sides(e["sides"])}')
            lines.append(f'- Looks like: {self._description(e)}')
            if e['status'] == 'done':
                if e.get('evidence'):
                    lines.append(f'- Evidence: {"; ".join(e["evidence"])}')
                lines.append(f'- Confidence {e["confidence"]:.2f} (clearly identified: {"yes" if e["relevant"] else "no"})')
                if e.get('image'):
                    lines.append(f'- Image: `{os.path.join(self.run_dir, e["image"])}`')
            lines.append('')
        if to_visit:
            lines += ['## Still to visit', '', ', '.join(to_visit), '']
        return '\n'.join(lines)

    def write(self, **render_args) -> str:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(self.render(**render_args))
        os.replace(tmp, self.path)          # the file is never half written
        return self.path


def within_reach(pose, centre, radius_m: float) -> bool:
    return pose is not None and math.hypot(pose[0] - centre[0], pose[1] - centre[1]) <= radius_m
