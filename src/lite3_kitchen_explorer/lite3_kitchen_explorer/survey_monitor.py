"""Print the environment reports as readable lines (used by the tmux status window).

    ros2 run lite3_kitchen_explorer survey_monitor
"""
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from std_msgs.msg import Bool, String

BOLD, CYAN, GREEN, YELLOW, RED, END = '\033[1m', '\033[36m', '\033[32m', '\033[33m', '\033[31m', '\033[0m'


def format_report(report):
    s = report['summary']
    pose = report.get('pose')
    where = f'({pose[0]:.2f}, {pose[1]:.2f})' if pose else '(no pose)'
    colour = GREEN if s['identified_views'] else YELLOW
    lines = [f'{BOLD}STOP {report["stop"]}{END} [{report["reason"]}] at {where} {report["wall_time"]}',
             f'  {colour}{BOLD}{s["environment"]}{END}  conf {s["confidence"]:.2f}, '
             f'{s["identified_views"]}/{s["total_views"]} views identified']
    if s.get('evidence'):
        lines.append(f'  evidence: {"; ".join(s["evidence"])}')
    if report.get('tile'):
        c = report.get('tile_counts', {})
        sides = ' '.join(f'{k}={v}' for k, v in (report.get('tile_sides') or {}).items())
        lines.append(f'  tile {report["tile"]} [{sides}]  visited {c.get("visited", 0)}, '
                     f'to visit {c.get("to_visit", 0)}')
    nxt = report.get('next_frontier')
    if nxt and nxt.get('kind') == 'tile':
        lines.append(f'  {CYAN}next: tile {nxt["tile"]} ({nxt["steps"]} step(s), {nxt["clearance_m"]:.2f} m from walls) '
                     f'chosen by {nxt["chosen_by"]} - {nxt.get("reason", "")}{END}')
    elif nxt and nxt.get('chosen_by') == 'AI':
        lines.append(f'  {CYAN}next: AI chose frontier {nxt["id"]} ({nxt["clearance_m"]:.2f} m from walls) '
                     f'- {nxt.get("reason", "")}{END}')
    elif nxt:
        lines.append(f'  {YELLOW}next: explorer decides ({nxt.get("note", "")}){END}')
    for v in report['views']:
        heading = f'{v.get("heading_deg", 0):6.0f}°'
        if 'error' in v:
            lines.append(f'  {RED}{heading}  query failed: {v["error"][:80]}{END}')
        else:
            mark = '✓' if v['relevant_to_mission'] else '·'
            lines.append(f'  {heading} {mark} {v["scene_description"]} ({v["confidence"]:.2f})')
    return '\n'.join(lines)


class SurveyMonitor(Node):
    def __init__(self):
        super().__init__('survey_monitor')
        latched = QoSProfile(depth=50, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/environment_report', self._on_report, latched)
        self.create_subscription(Bool, '/exploration_complete', self._on_complete,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/tile_map', self._on_tiles,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.tile_announced = False
        self.last_trail_len = 0
        self.seen = set()
        print(f'{CYAN}Waiting for environment reports on /environment_report ...{END}', flush=True)

    def _on_report(self, msg):
        try:
            report = json.loads(msg.data)
        except ValueError:
            print(msg.data, flush=True)
            return
        if report.get('stop') in self.seen:
            return
        self.seen.add(report.get('stop'))
        print(format_report(report) + '\n', flush=True)

    def _on_tiles(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        init = data.get('initial_tile') or {}
        if not self.tile_announced and init:
            self.tile_announced = True
            print(f'{CYAN}{BOLD}INITIAL TILE{END} {init.get("placement")} ({init.get("kind")}, {init.get("how")}); '
                  f'grid yaw {init.get("grid", {}).get("yaw_deg")} deg\n'
                  f'  sides from map: {init.get("map_sides")}  from AI: {init.get("ai_sides") or "-"}\n'
                  f'{CYAN}Exploration started.{END}\n', flush=True)
        trail = data.get('trail', [])
        if len(trail) > self.last_trail_len:
            self.last_trail_len = len(trail)
            c = data.get('counts', {})
            print(f'  trail: {" -> ".join(trail[-8:])}   (visited {c.get("visited", 0)}, '
                  f'to visit {c.get("to_visit", 0)})', flush=True)

    def _on_complete(self, msg):
        if msg.data:
            print(f'{GREEN}{BOLD}=== EXPLORATION COMPLETE: every frontier visited, '
                  f'{len(self.seen)} stop(s) reported ==={END}\n', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = SurveyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
