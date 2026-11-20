"""Print the tile reports (what each tile looks like) as readable lines (used by the tmux status window).

    ros2 run lite3_explorer survey_monitor
"""
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from std_msgs.msg import Bool, String

BOLD, CYAN, GREEN, YELLOW, RED, END = '\033[1m', '\033[36m', '\033[32m', '\033[33m', '\033[31m', '\033[0m'


def format_report(entry):
    """One tile report entry (/tile_report) as readable lines."""
    pose = entry.get('pose')
    where = f'({pose[0]:.2f}, {pose[1]:.2f})' if pose else '(no pose)'
    head = f'{BOLD}TILE {entry["tile"]}{END} #{entry["order"]} [{entry.get("source", "")}] at {where} {entry.get("reached_at", "")}'
    sides = ' '.join(f'{k}={v}' for k, v in (entry.get('sides') or {}).items())
    lines = [head, f'  map: {sides or "-"}']
    if entry['status'] == 'done':
        colour = GREEN if entry.get('relevant') else YELLOW
        lines.append(f'  {colour}{BOLD}{entry["description"]}{END}  conf {entry["confidence"]:.2f}')
        if entry.get('evidence'):
            lines.append(f'  evidence: {"; ".join(entry["evidence"])}')
    elif entry['status'] == 'error':
        lines.append(f'  {RED}AI description failed: {str(entry.get("error", ""))[:100]}{END}')
    else:
        lines.append(f'  {YELLOW}{entry.get("error", entry["status"])}{END}')
    return '\n'.join(lines)


class SurveyMonitor(Node):
    def __init__(self):
        super().__init__('survey_monitor')
        latched = QoSProfile(depth=50, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/tile_report', self._on_report, latched)
        self.create_subscription(Bool, '/exploration_complete', self._on_complete,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/tile_map', self._on_tiles,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.tile_announced = False
        self.last_trail_len = 0
        self.seen = set()
        print(f'{CYAN}Waiting for tile reports on /tile_report ...{END}', flush=True)

    def _on_report(self, msg):
        try:
            report = json.loads(msg.data)
        except ValueError:
            print(msg.data, flush=True)
            return
        self.seen.add(report.get('tile'))
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
                  f'  sides from map: {init.get("map_sides")}\n'
                  f'{CYAN}Exploration started.{END}\n', flush=True)
        trail = data.get('trail', [])
        if len(trail) > self.last_trail_len:
            self.last_trail_len = len(trail)
            c = data.get('counts', {})
            print(f'  trail: {" -> ".join(trail[-8:])}   (visited {c.get("visited", 0)}, '
                  f'to visit {c.get("to_visit", 0)}, unreachable {c.get("unreachable", 0)})', flush=True)

    def _on_complete(self, msg):
        if msg.data:
            print(f'{GREEN}{BOLD}=== EXPLORATION COMPLETE: '
                  f'{len(self.seen)} tile(s) reported ==={END}\n', flush=True)


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
