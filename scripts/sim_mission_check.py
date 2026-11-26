#!/usr/bin/env python3
"""Check a sim run (scripts/sim_mission_check.sh) against the maze's ground truth: every mission answer, every room.

    python3 scripts/sim_mission_check.py <maze.yaml> <run output dir>

Expected answers come from the maze's `perception:` block (lite3_sim/sim_perception.py) and its scene tiles:
  survivor with qr -> M03 + M04 (the QR text),  survivor without -> M05,  object hand_tools -> M01,  hazard -> M02,
each with the scene type of the maze tile it stands on. Also checked: every named scene tile got its scene type from
the AI stand-in (/scene/result via mission_manager events), no ordinary tile was named a room, nothing was sent twice,
and no node crashed. Exit code 0 = all passed.
"""
import glob
import json
import os
import sys

import yaml

from lite3_sim.maze import load_maze

TASK_OF = {'hand_tools': 'M01'}


def main():
    maze_path, out = sys.argv[1], os.path.expanduser(sys.argv[2])
    maze = load_maze(maze_path)
    spec = (yaml.safe_load(open(maze_path)) or {}).get('perception') or {}
    room = lambda t: maze.scene_types.get(tuple(t), '')          # noqa: E731

    expected = {}                                                # task id -> (scene_type, message)
    for s in spec.get('survivors') or []:
        expected['M03' if s.get('qr') else 'M05'] = (room(s['tile']), '')
        if s.get('qr'):
            expected['M04'] = (None, str(s['qr']))               # M04: only the message is scored
    for o in spec.get('objects') or []:
        tid = 'M02' if o.get('type') == 'hazard' else TASK_OF.get(o['label'])
        if tid:
            expected[tid] = (room(o['tile']), '')

    results = []
    p = os.path.join(out, 'results.jsonl')
    if os.path.exists(p):
        results = [json.loads(line) for line in open(p) if line.strip() and 'task_id' in line
                   and 'server_reply_for' not in line]
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    print(f'== mission answers ({len(results)} sent, {out}/results.jsonl)')
    by_task = {}
    for r in results:
        by_task.setdefault(r['task_id'], []).append(r)
    for tid in sorted(set(expected) | set(by_task)):
        got = by_task.get(tid, [])
        if tid not in expected:
            check(False, f'{tid}: sent but nothing in the maze should answer it: {got}')
            continue
        scene, msg = expected[tid]
        if not got:
            check(False, f'{tid}: never sent (expected scene {scene or "-"}{", message " + repr(msg) if msg else ""})')
            continue
        r = got[0]
        check(len(got) == 1, f'{tid}: sent {len(got)} time(s)')
        if scene is not None:
            check(r.get('scene_type') == scene, f'{tid}: scene_type {r.get("scene_type")!r} (expected {scene!r}), '
                                                f'pos {r.get("pos")}')
        if msg:
            check(r.get('message') == msg, f'{tid}: message {r.get("message")!r}')

    # the mission manager's run folder: newest run_* started after this check's output dir was made
    runs = sorted(glob.glob(os.path.join(out, 'run_*', 'events.jsonl')), key=os.path.getmtime)   # this run's only
    print(f'== scene tiles ({runs[-1] if runs else "no mission events found"})')
    if runs:
        ev = [json.loads(line) for line in open(runs[-1]) if line.strip()]
        named = {}
        for e in ev:
            if e.get('kind') == 'scene_described' or e.get('kind') == 'scene_traversable':
                named.setdefault(e.get('tile'), []).append(e.get('kind'))
        scene_res = [json.loads(line) for line in open(os.path.join(out, 'scene_results.jsonl'))] \
            if os.path.exists(os.path.join(out, 'scene_results.jsonl')) else []
        names = {}
        for s in scene_res:
            names.setdefault(s.get('scene_type'), []).append(s.get('tile'))
        for t, n in sorted(maze.scene_types.items()):
            check(n in names, f'maze scene tile {t[0]},{t[1]} {n}: named on /scene/result as tiles {names.get(n)}')
        wrong = {k: v for k, v in names.items() if k not in set(maze.scene_types.values()) | {'traversable', None, ''}}
        check(not wrong, f'no room named that is not in the maze: {wrong or "none"}')
        looks = [e.get('tile') for e in ev if e.get('kind') == 'scene']
        props = [(e.get('tile'), e.get('frames')) for e in ev if e.get('kind') == 'prop_scene']
        print(f'  scene looks: {looks}')
        print(f'  prop votes: {props}')
        reached = [e.get('tile') for e in ev if e.get('kind') == 'tile_reached']
        print(f'  tiles reached: {reached}')
        modes = [e.get('mode') for e in ev if e.get('kind') == 'mode']
        print(f'  last mode: {modes[-1] if modes else "-"}')

    print('== crashes')
    for log in sorted(glob.glob(os.path.join(out, '*.log'))):
        txt = open(log, errors='replace').read()
        cut = txt.find("sending signal 'SIGINT'")          # the launch's Ctrl-C at the end: what dies after it is shutdown
        txt = txt if cut < 0 else txt[:cut]
        lines = txt.splitlines()
        bad = [l for k, l in enumerate(lines) if ('Traceback' in l or 'process has died' in l)
               and not any(x in ' '.join(lines[k:k + 25]) for x in ('KeyboardInterrupt', 'ExternalShutdownException'))]
        check(not bad, f'{os.path.basename(log)}: {bad[:3] if bad else "no tracebacks / dead processes"}')
    print('ALL PASSED' if ok else 'SOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
