"""Check a maze file before simulating it: parse it, print a summary and redraw it from the parsed edges.

    ros2 run lite3_sim maze_preview ~/my_maze.yaml
    ros2 run lite3_sim maze_preview ~/my_maze.yaml --png maze.png        # needs matplotlib

The redraw uses tile coordinates: the column numbers along the top are i, the row numbers on the left are j.
If it does not look like your layout, a character is probably off by one column (tiles are 4 characters wide).
"""
from __future__ import annotations

import argparse
import math
import sys

from lite3_sim.maze import CURTAIN, SLIT, STAIRS, WALL, MazeError, load_maze

H = {WALL: '---', CURTAIN: '~~~', SLIT: '===', None: '   '}
V = {WALL: '|', CURTAIN: '~', SLIT: '!', None: ' '}


def render(maze) -> str:
    sx, sy, syaw = maze.start
    start = (int(sx // maze.tile_size), int(sy // maze.tile_size))
    arrow = {0: '>', 90: '^', 180: '<', -180: '<', -90: 'v'}.get(int(round(math.degrees(syaw) / 90.0)) * 90, 'S')
    lines = ['     ' + ''.join(f'{i:^4d}' for i in range(maze.cols))]
    for j in range(maze.rows - 1, -1, -1):
        top = '    +' + ''.join(H[maze.edge((i, j), 'N')] + '+' for i in range(maze.cols))
        row = f'{j:3d} ' + V[maze.edge((0, j), 'W')]
        for i in range(maze.cols):
            cell = ('#' if (i, j) in maze.solid else '0' if (i, j) in maze.scenes else 'M' if (i, j) in maze.stairs
                    else arrow if (i, j) == start
                    else maze.labels.get((i, j), ''))
            row += f'{cell:^3s}' + V[maze.edge((i, j), 'E')]
        lines += [top, row]
    lines.append('    +' + ''.join(H[maze.edge((i, 0), 'S')] + '+' for i in range(maze.cols)))
    return '\n'.join(lines)


def save_png(maze, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    w, h = maze.size_m
    fig, ax = plt.subplots(figsize=(max(4, w * 1.2), max(3, h * 1.2)))
    style = {WALL: dict(color='0.25', lw=3), SLIT: dict(color='tab:blue', lw=3), CURTAIN: dict(color='c', lw=3, ls='--'),
             STAIRS: dict(color='gold', lw=3, ls=':')}
    for s in maze.segments:
        ax.plot([s.x0, s.x1], [s.y0, s.y1], **style[s.kind])
    for t in maze.solid:
        ax.add_patch(plt.Rectangle((t[0] * maze.tile_size, t[1] * maze.tile_size), maze.tile_size, maze.tile_size,
                                   color='0.7'))
    for t in maze.scenes:
        ax.add_patch(plt.Rectangle((t[0] * maze.tile_size + 0.15, t[1] * maze.tile_size + 0.15),
                                   maze.tile_size - 0.3, maze.tile_size - 0.3, color='orange', alpha=0.5))
        ax.text(*maze.tile_centre(t), 'scene', ha='center', va='center', fontsize=8)
    for t, sides in maze.stairs.items():
        ax.add_patch(plt.Rectangle((t[0] * maze.tile_size, t[1] * maze.tile_size), maze.tile_size, maze.tile_size,
                                   color='gold', alpha=0.35))
        ax.text(*maze.tile_centre(t), 'stairs\n(' + '/'.join(sides) + ')', ha='center', va='center', fontsize=8)
    for t, text in maze.labels.items():
        ax.text(*maze.tile_centre(t), text, ha='center', va='center')
    sx, sy, syaw = maze.start
    ax.arrow(sx, sy, 0.3 * math.cos(syaw), 0.3 * math.sin(syaw), width=0.04, color='g')
    for i in range(maze.cols):
        for j in range(maze.rows):
            ax.text((i + 0.08) * maze.tile_size, (j + 0.08) * maze.tile_size, f'{i},{j}', fontsize=7, color='0.5')
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.2, h + 0.2)
    ax.set_title(f'{maze.name}: walls (grey), slit panels (blue), curtains (cyan dashed), scene tiles (orange), '
                 f'staircases (gold, risers dotted)')
    fig.savefig(path, dpi=120, bbox_inches='tight')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('maze')
    ap.add_argument('--png', default='', help='also save a picture (needs matplotlib)')
    a = ap.parse_args(argv)
    try:
        maze = load_maze(a.maze)
    except (MazeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(maze.summary())
    print(render(maze))
    unreachable = sorted(set(maze.floor_tiles()) - maze.reachable_tiles() - set(maze.stairs))
    if unreachable:
        print(f'Floor tiles the dog can never reach (closed off): {unreachable}')
    for t, sides in sorted(maze.stairs.items()):
        print(f'Staircase on tile {t[0]},{t[1]}: climbed from {", ".join(sides)} ({maze.stair_steps} steps of '
              f'{maze.stair_rise_m * 100:.0f} cm up to the landing); the stairs mode crosses it')
    beyond = sorted(maze.reachable_tiles() - maze.reachable_tiles(through_stairs=False))
    if beyond:
        print(f'Only reachable over a staircase: {beyond}')
    if a.png:
        save_png(maze, a.png)
        print(f'saved {a.png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
