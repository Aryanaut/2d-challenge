"""The AI tasks, without ROS: build the context (camera frame, lidar map, lidar ranges), query the server, read the answer.

Every task takes a JSON-able request dict and returns a JSON-able result dict, so a maze explorer only has to
send {"task": ..., ...} and read the reply. The ROS node (llm_client.py) feeds this class the live camera frames,
the latest /map and the latest /scan, and moves the dicts over /llm/request and /llm/response.

Tasks
  capture_view         keep the next camera frame (+ the lidar ranges at that moment) under a name
  describe_views       send the kept frames of one stop to the AI: what kind of place is this?
  describe_tile        (the node keeps the next camera frame first) report what one tile looks like, from that
                       frame, the lidar ranges and the tile context the explorer sends
  place_initial_tile   AI picks the best of the proposed initial 1 m tile placements (map image)
  choose_target        AI picks the next tile (mode "tile") or frontier (mode "frontier") from the map image,
                       and may give 2-3 waypoints with a yaw for the way there
  render_map           draw the map (tiles, trail, route, ...) to a PNG; no AI call
  ping                 liveness
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from lite3_llm_tools import prompts as P
from lite3_llm_tools.gridmap import GridInfo
from lite3_llm_tools.image_utils import fit_within
from lite3_llm_tools.map_render import render_initial_tile_panels, render_map
from lite3_llm_tools.perception_client import PerceptionClient, PerceptionError, read_prompt
from lite3_llm_tools.tile_view import TileView, parse_tile_label

MapSnapshot = Tuple[np.ndarray, GridInfo]


class TaskError(Exception):
    """A request that cannot be served (bad request, no map yet, ...). The message goes back to the caller."""


@dataclass
class Settings:
    output_dir: str = '~/lite3_survey_runs'
    query_retries: int = 1                 # extra attempts per view (the client also retries connect errors)
    error_backoff_s: float = 5.0
    view_query_workers: int = 3            # views of one stop sent at once
    wall_band_m: float = 0.35              # red band drawn next to walls, and told to the AI
    min_clearance_m: float = 0.30
    lidar_context: bool = True             # add the lidar ranges to the prompts
    max_captures: int = 64
    max_upload_w: int = 1024               # the server rejects larger images (HTTP 413 image_dimensions_exceeded)
    max_upload_h: int = 768


def _png(img, max_w: int = 0, max_h: int = 0) -> bytes:
    ok, buf = cv2.imencode('.png', fit_within(img, max_w, max_h))
    if not ok:
        raise TaskError('PNG encoding failed')
    return buf.tobytes()


class LlmTools:
    def __init__(self, client: PerceptionClient, settings: Optional[Settings] = None,
                 prompt_overrides: Optional[Dict[str, str]] = None,
                 get_map: Callable[[], Optional[MapSnapshot]] = lambda: None,
                 get_scan_text: Callable[[], Optional[str]] = lambda: None,
                 sleep: Callable[[float], None] = time.sleep):
        self.client = client
        self.s = settings or Settings()
        overrides = prompt_overrides or {}
        self.templates = {task: read_prompt(P.prompt_path(task, overrides.get(task, ''))) for task in P.PROMPT_FILES}
        self.get_map = get_map
        self.get_scan_text = get_scan_text
        self._sleep = sleep
        self._lock = threading.Lock()
        self.captures: Dict[Tuple[str, str], dict] = {}        # (save_dir, name) -> kept frame
        self.default_dir = os.path.join(os.path.expanduser(self.s.output_dir),
                                        'run_' + time.strftime('%Y%m%d_%H%M%S'))
        self.queries = 0
        self.last_error = ''

    # ------------------------------------------------------------------ plumbing
    def handle(self, request: dict) -> dict:
        """Serve one AI request (blocking). Always returns {"id", "task", "ok", "result" | "error", "elapsed_s"}."""
        started = time.monotonic()
        task = request.get('task')
        reply: Dict[str, Any] = {'id': request.get('id'), 'task': task}
        handler = {
            'describe_views': self.describe_views, 'describe_tile': self.describe_tile, 'place_initial_tile': self.place_initial_tile,
            'choose_target': self.choose_target, 'render_map': self.render_map_task,
            'ping': lambda req: {'pong': True},
        }.get(task)
        try:
            if handler is None:
                raise TaskError(f'unknown task {task!r}')
            reply['result'] = handler(request)
            reply['ok'] = True
        except (TaskError, PerceptionError) as exc:
            reply['ok'], reply['error'] = False, str(exc)
            self.last_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - one bad request must never kill the node
            reply['ok'], reply['error'] = False, f'{type(exc).__name__}: {exc}'
            self.last_error = reply['error']
        reply['elapsed_s'] = round(time.monotonic() - started, 2)
        return reply

    def _dir(self, request: dict) -> str:
        path = os.path.expanduser(request.get('save_dir') or self.default_dir)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _write(path: str, data: bytes):
        with open(path, 'wb') as fh:
            fh.write(data)

    @staticmethod
    def _save_json(path: str, data):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    def _with_lidar(self, prompt: str, scan_text: Optional[str] = None) -> str:
        text = scan_text if scan_text is not None else (self.get_scan_text() if self.s.lidar_context else None)
        return f'{prompt}\n\n{text}' if text and self.s.lidar_context else prompt

    def _map(self) -> MapSnapshot:
        snap = self.get_map()
        if snap is None:
            raise TaskError('no map yet')
        return snap

    # ------------------------------------------------------------------ camera views
    def store_capture(self, name: str, jpeg: bytes, save_dir: str = '', pose=None, scan_text: Optional[str] = None):
        """Called by the node when the frame for a capture_view request arrives."""
        folder = self._dir({'save_dir': save_dir})
        self._write(os.path.join(folder, name + '.jpg'), jpeg)
        with self._lock:
            self.captures[(folder, name)] = {'jpeg': jpeg, 'pose': pose, 'scan_text': scan_text}
            while len(self.captures) > self.s.max_captures:
                self.captures.pop(next(iter(self.captures)))
        return {'name': name, 'image': name + '.jpg', 'lidar': scan_text}

    def describe_views(self, request: dict) -> dict:
        folder = self._dir(request)
        views = request.get('views')
        if not isinstance(views, list) or not views:
            raise TaskError('describe_views needs a non-empty "views" list')
        with self._lock:
            kept = [self.captures.pop((folder, v.get('name')), None) for v in views]
        workers = max(1, min(self.s.view_query_workers, len(views)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(lambda args: self._describe_one(folder, *args), zip(views, kept)))
        return {'views': out}

    def describe_tile(self, request: dict) -> dict:
        """One tile report. The node has already kept the camera frame under request["name"] (store_capture)."""
        folder = self._dir(request)
        name = str(request.get('name') or '')
        if not name:
            raise TaskError('describe_tile needs "name"')
        with self._lock:
            capture = self.captures.pop((folder, name), None)
        view = self._describe_one(folder, {'name': name}, capture, task='describe_tile', context=request.get('context', ''))
        if 'error' in view:
            raise PerceptionError(view['error'])
        return {'tile': request.get('tile'), **{k: v for k, v in view.items() if k != 'name'}}

    def _describe_one(self, folder: str, view: dict, capture: Optional[dict], task: str = 'describe_view',
                      context: str = '') -> dict:
        name = str(view.get('name'))
        meta = {k: view[k] for k in ('heading_deg', 'relative_heading_deg') if k in view}
        if capture is None:
            return {'name': name, 'image': name + '.jpg', 'error': 'no captured frame under this name', **meta}
        prompt = self._with_lidar(self.templates[task], capture.get('scan_text'))
        if context:
            prompt += '\n\n' + context
        error = ''
        for attempt in range(self.s.query_retries + 1):
            try:
                result = self.client.analyze(capture['jpeg'], name + '.jpg', 'image/jpeg', prompt)
            except PerceptionError as exc:
                error = str(exc)
                self._save_json(os.path.join(folder, name + '.json'), {'error': error, 'pose': capture.get('pose')})
                if attempt < self.s.query_retries:
                    self._sleep(max(self.s.error_backoff_s, exc.retry_after_s or 0.0))
                continue
            self.queries += 1
            answer = {**result.as_dict(), 'image': name + '.jpg', 'elapsed_s': round(result.elapsed_s, 2), **meta}
            self._save_json(os.path.join(folder, name + '.json'),
                            {**answer, 'pose': capture.get('pose'), 'lidar': capture.get('scan_text'), 'raw': result.raw})
            return answer
        return {'name': name, 'image': name + '.jpg', 'error': error, **meta}

    # ------------------------------------------------------------------ map based choices
    def place_initial_tile(self, request: dict) -> dict:
        grid, info = self._map()
        folder = self._dir(request)
        name = request.get('name') or 'initial_tile'
        pose = request.get('pose')
        cands = request.get('candidates')
        if not pose or not isinstance(cands, list) or not cands:
            raise TaskError('place_initial_tile needs "pose" and "candidates"')
        views = [{**c, 'grid': TileView.from_dict(c['grid'])} for c in cands]
        img = render_initial_tile_panels(grid, info, pose, views,
                                         wall_band_m=request.get('wall_band_m', self.s.wall_band_m))
        png = _png(img, self.s.max_upload_w, self.s.max_upload_h)      # saved = exactly what the AI sees
        self._write(os.path.join(folder, name + '.png'), png)
        prompt = self._with_lidar(P.build_initial_tile_prompt(
            self.templates['place_initial_tile'], cands, float(request.get('size_m', 1.0))))
        self._save_json(os.path.join(folder, name + '_prompt.json'), {'prompt': prompt, 'candidates': cands})
        result = self.client.analyze(png, name + '.png', 'image/png', prompt)
        self.queries += 1
        answer = result.as_dict()
        self._save_json(os.path.join(folder, name + '_answer.json'), {**answer, 'raw': result.raw})
        choice, ai_sides = P.parse_tile_choice(result.scene_description, result.evidence, {str(c['id']) for c in cands})
        return {'answer': answer, 'choice': choice, 'ai_sides': ai_sides, 'image': name + '.png',
                'elapsed_s': round(result.elapsed_s, 2)}

    @staticmethod
    def _tiles_dict(raw) -> Dict[Tuple[int, int], dict]:
        """[{"tile": [i, j], "status": "visited"}, ...] (or {"i,j": {...}}) -> {(i, j): {"status": ...}}."""
        out = {}
        items = raw.items() if isinstance(raw, dict) else ((t.get('tile'), t) for t in (raw or []))
        for label, info in items:
            tile = parse_tile_label(label)
            if tile is not None:
                out[tile] = dict(info)
        return out

    def _render(self, request: dict, candidates: List[dict], route=()):
        grid, info = self._map()
        tile_grid = TileView.from_dict(request['tile_grid']) if request.get('tile_grid') else None
        trail = [t for t in (parse_tile_label(x) for x in request.get('trail', [])) if t is not None]
        return render_map(grid, info, request.get('pose'), candidates, request.get('visited', []),
                          wall_band_m=request.get('wall_band_m', self.s.wall_band_m),
                          tile_grid=tile_grid, tiles=self._tiles_dict(request.get('tiles')), trail=trail,
                          route=route)

    def choose_target(self, request: dict) -> dict:
        mode = request.get('mode')
        cands = request.get('candidates')
        if mode not in ('tile', 'frontier') or not isinstance(cands, list) or not cands:
            raise TaskError('choose_target needs "mode" ("tile" or "frontier") and "candidates"')
        folder = self._dir(request)
        name = request.get('name') or 'choice'
        try:
            img = self._render(request, cands)
        except ValueError as exc:
            raise TaskError(f'map render failed: {exc}') from exc
        png = _png(img, self.s.max_upload_w, self.s.max_upload_h)
        self._write(os.path.join(folder, name + '.png'), png)
        environment = request.get('environment') or 'unknown'
        stops = int(request.get('stops_so_far', 1))
        summary = request.get('tile_summary', '')
        if mode == 'tile':
            prompt = P.build_tile_prompt(self.templates['choose_tile'], cands, environment,
                                         request.get('min_clearance_m', self.s.min_clearance_m), stops, summary)
        else:
            prompt = P.build_prompt(self.templates['choose_frontier'], cands, environment,
                                    request.get('wall_band_m', self.s.wall_band_m), stops, summary)
        prompt = self._with_lidar(prompt)
        self._save_json(os.path.join(folder, name + '_prompt.json'), {'prompt': prompt, 'mode': mode, 'candidates': cands})
        result = self.client.analyze(png, name + '.png', 'image/png', prompt)
        self.queries += 1
        answer = result.as_dict()
        self._save_json(os.path.join(folder, name + '_answer.json'), {**answer, 'raw': result.raw})
        choice = P.parse_choice(result.scene_description, result.evidence, [c['id'] for c in cands])
        return {'answer': answer, 'choice': choice, 'extras': result.extras, 'image': name + '.png',
                'elapsed_s': round(result.elapsed_s, 2)}

    def render_map_task(self, request: dict) -> dict:
        folder = self._dir(request)
        name = request.get('name') or 'map'
        route = [tuple(p) for p in request.get('route', [])]
        try:
            img = self._render(request, request.get('candidates', []), route)
        except ValueError as exc:
            raise TaskError(f'map render failed: {exc}') from exc
        self._write(os.path.join(folder, name + '.png'), _png(img))
        return {'image': name + '.png'}
