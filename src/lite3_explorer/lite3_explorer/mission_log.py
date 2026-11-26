"""Perception sightings, mission reporter results and their log lines, for the tile report (no ROS).

The mission manager feeds it from the perception / qr_scanner / mission reporter topics and /rosout. Nothing arrives
when those nodes are not running, and then the report has no such sections. Kept for posterity in the run folder
too (appended as they arrive, so a crash loses nothing):
  run_dir/sightings.jsonl         one line per QR / hazard / survivor / object sighting (when + tile)
  run_dir/mission_results.jsonl   one line per /mission/result (what the reporter sent)
  run_dir/perception.log, run_dir/mission_reporter.log   their /rosout lines, complete
The report shows every sighting and result, and the last `log_tail` log lines of each group.
"""
import datetime as dt
import json
import os
import time
from collections import deque
from typing import Dict, List, Optional

LEVELS = {10: 'DEBUG', 20: 'INFO', 30: 'WARN', 40: 'ERROR', 50: 'FATAL'}
KIND_ORDER = ('qr', 'hazard', 'survivor', 'object')
GROUP_TITLES = {'perception': 'Perception log', 'reporter': 'Mission reporter log'}


def kind_of(event: dict) -> str:
    """/perception/event -> qr | hazard | survivor | object."""
    if str(event.get('label', '')).lower() == 'survivor':
        return 'survivor'
    t = str(event.get('type', 'object')).lower()
    return t if t in ('qr', 'hazard') else 'object'


def wall(t) -> float:
    """A message stamp as wall time; a sim-clock (or missing) stamp -> now."""
    now = time.time()
    return float(t) if isinstance(t, (int, float)) and abs(now - float(t)) < 3600.0 else now


def _md(text) -> str:
    return str(text).replace('|', '/').replace('\n', ' ').strip()


def _xy(p) -> str:
    return f'({p[0]:.2f}, {p[1]:.2f})' if p else '-'


class MissionLog:
    def __init__(self, run_dir: str, t0_wall: Optional[float] = None, log_tail: int = 300):
        self.run_dir = run_dir
        self.t0 = t0_wall
        self.sightings: List[dict] = []
        self.results: List[dict] = []
        self.log_tail = int(log_tail)
        self.logs: Dict[str, deque] = {}
        self.log_counts: Dict[str, int] = {}
        self._qr_texts = set()

    # ------------------------------------------------------------------ inputs
    def _stamp(self, t: float) -> dict:
        d = {'time': dt.datetime.fromtimestamp(t).isoformat(timespec='seconds')}
        if self.t0 is not None:
            d['mission_s'] = round(t - self.t0, 1)
        return d

    def _append(self, name: str, line: str):
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, name), 'a', encoding='utf-8') as fh:
                fh.write(line + '\n')
        except OSError:
            pass

    def sighting(self, t: float, kind: str, label: str, *, confidence=None, tile: Optional[str] = None,
                 pos=None, robot_tile: Optional[str] = None, robot_pos=None, source: str = '',
                 note: str = '') -> Optional[dict]:
        """One detection. tile/pos = where the thing is (its map position when known, else where the robot was).
        QR codes are kept once per text (qr_scanner and perception's own decoder both report them). The same kind +
        label on the same tile again only counts up `seen` / `last_time` of the first row (returns None)."""
        t = wall(t)
        if kind == 'qr':
            if label in self._qr_texts:
                return None
            self._qr_texts.add(label)
        for old in self.sightings:
            if old['kind'] == kind and old['label'] == label and old['tile'] == tile and tile is not None:
                old['seen'] += 1
                old['last_time'] = self._stamp(t)['time']
                return None
        s = {**self._stamp(t), 'kind': kind, 'label': label, 'confidence': confidence, 'tile': tile,
             'pos': [round(float(v), 2) for v in pos[:2]] if pos else None, 'robot_tile': robot_tile,
             'robot_pos': [round(float(v), 2) for v in robot_pos[:2]] if robot_pos else None,
             'source': source, 'note': note, 'seen': 1}
        self.sightings.append(s)
        self._append('sightings.jsonl', json.dumps(s, ensure_ascii=False))
        return s

    def fill_tiles(self, tile_at) -> bool:
        """Sightings made before the tile grid existed get their tiles now (tile_at(xy) -> "i,j" or None)."""
        changed = False
        for s in self.sightings:
            for key, pos in (('tile', 'pos'), ('robot_tile', 'robot_pos')):
                if s.get(key) is None and s.get(pos):
                    s[key] = tile_at(s[pos])
                    changed = changed or s[key] is not None
        return changed

    def result(self, t: float, data: dict) -> dict:
        r = {**self._stamp(wall(t)), **data}
        self.results.append(r)
        self._append('mission_results.jsonl', json.dumps(r, ensure_ascii=False))
        return r

    def log_line(self, group: str, t: float, level: int, node: str, msg: str):
        t = wall(t)
        line = f'[{dt.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]}] [{LEVELS.get(level, level)}] [{node}] {msg}'
        self.logs.setdefault(group, deque(maxlen=max(self.log_tail, 1))).append(line)
        self.log_counts[group] = self.log_counts.get(group, 0) + 1
        self._append({'perception': 'perception.log', 'reporter': 'mission_reporter.log'}.get(group, f'{group}.log'), line)

    @property
    def empty(self) -> bool:
        return not (self.sightings or self.results or self.logs)

    # ------------------------------------------------------------------ output
    def render(self) -> List[str]:
        if self.empty:
            return []
        lines = []
        if self.sightings or 'perception' in self.logs:
            lines += ['## Detections (QR / hazard / survivor)', '',
                      'When and where each was first detected (the same thing on the same tile again is counted on its row). Tile = the tile the thing is on (its map position from '
                      'depth / the QR locator; without one, the robot\'s tile). Full list: '
                      f'`{os.path.join(self.run_dir, "sightings.jsonl")}`.', '']
            for kind in KIND_ORDER:
                n = sum(1 for s in self.sightings if s['kind'] == kind)
                lines.append(f'- **{kind}:** {n}' + ('' if n else ' (none detected)'))
            lines += ['', '| # | Time | Mission t | Kind | Label / text | Conf. | Tile | Position | Robot tile | Robot pos | Source |',
                      '|---|---|---|---|---|---|---|---|---|---|---|']
            for k, s in enumerate(self.sightings, 1):
                conf = f'{s["confidence"]:.2f}' if isinstance(s.get('confidence'), (int, float)) else '-'
                mt = f'{s["mission_s"]:.0f} s' if s.get('mission_s') is not None else '-'
                label = (s['label'] or '-') + (f' ({s["note"]})' if s.get('note') else '')
                if s.get('seen', 1) > 1:
                    label += f' - seen {s["seen"]}x, last {s["last_time"][11:]}'
                lines.append(f'| {k} | {s["time"][11:]} | {mt} | {s["kind"]} | {_md(label)} | {conf} | '
                             f'{s["tile"] or "-"} | {_xy(s["pos"])} | {s["robot_tile"] or "-"} | {_xy(s["robot_pos"])} | '
                             f'{_md(s["source"])} |')
            lines.append('')
        if self.results or 'reporter' in self.logs:
            lines += ['## Mission reporter results', '',
                      f'Every result the reporter sent (/mission/result). Also in '
                      f'`{os.path.join(self.run_dir, "mission_results.jsonl")}`.', '']
            if self.results:
                lines += ['| Time | Mission t | Task | Type | Scene | Label | Pos | Message |', '|---|---|---|---|---|---|---|---|']
                for r in self.results:
                    mt = f'{r["mission_s"]:.0f} s' if r.get('mission_s') is not None else '-'
                    lines.append(f'| {r["time"][11:]} | {mt} | {_md(r.get("task_id", "-"))} | {_md(r.get("task_type", "-"))} | '
                                 f'{_md(r.get("scene_type") or "-")} | {_md(r.get("label") or "-")} | '
                                 f'{_xy(r.get("pos")) if r.get("pos") else "null"} | {_md(r.get("message") or "")} |')
            else:
                lines.append('(nothing sent yet)')
            lines.append('')
        for group in ('perception', 'reporter'):
            if group not in self.logs:
                continue
            total, shown = self.log_counts[group], len(self.logs[group])
            name = {'perception': 'perception.log', 'reporter': 'mission_reporter.log'}[group]
            lines += [f'## {GROUP_TITLES[group]}', '',
                      f'{total} line(s)' + (f', last {shown} shown' if shown < total else '') +
                      f'. Complete: `{os.path.join(self.run_dir, name)}`.', '',
                      '<details><summary>show</summary>', '', '```', *self.logs[group], '```', '', '</details>', '']
        return lines
