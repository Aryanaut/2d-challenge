"""Raw single-key input from a terminal (Linux / SSH), including arrow keys."""

import os
import select
import sys

ARROWS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}


def parse_keys(data: str):
    """Split raw terminal input into key names: letters as-is, arrows -> UP/DOWN/LEFT/RIGHT,
    a lone Escape -> ESC. Unknown escape sequences are skipped."""
    keys, i = [], 0
    while i < len(data):
        ch = data[i]
        if ch == "\x1b":
            if data[i + 1:i + 2] in ("[", "O") and i + 2 < len(data):
                # ESC [ <params: digits ; > <final byte>, e.g. ESC[A or ESC[1;5C (ctrl+right)
                j = i + 2
                while j < len(data) and (data[j].isdigit() or data[j] == ";"):
                    j += 1
                if j < len(data):
                    if data[j] in ARROWS:
                        keys.append(ARROWS[data[j]])
                    i = j + 1
                else:
                    i = j               # incomplete sequence: drop it
                continue
            keys.append("ESC")
            i += 1
            continue
        if ch in ("\r", "\n"):
            i += 1
            continue
        keys.append(ch)
        i += 1
    return keys


class RawKeyboard:
    """Context manager: puts the terminal in cbreak mode (Ctrl+C still works) and restores it."""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("the console needs an interactive terminal: start it with "
                               "`ros2 run lite3_teleop_vlm teleop_console`, not from a launch file")
        import termios
        import tty
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self, timeout: float):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return []
        data = os.read(self.fd, 64).decode(errors="ignore")
        return parse_keys(data)

    def __exit__(self, *exc):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        return False
