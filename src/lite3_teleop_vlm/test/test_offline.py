"""Offline tests (no ROS, no robot): python3 -m pytest test/ -q"""

import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lite3_teleop_vlm import gateway  # noqa: E402
from lite3_teleop_vlm.broker_queue import Request, RequestQueue  # noqa: E402
from lite3_teleop_vlm.keyboard import parse_keys  # noqa: E402
from lite3_teleop_vlm.teleop_logic import Teleop, TeleopConfig  # noqa: E402


# ---------- driving ----------
def test_hold_to_move_and_deadman():
    t = Teleop(TeleopConfig(linear_speed=0.3, angular_speed=0.6, hold_s=0.6))
    assert t.key("w", 0.0) is None
    assert t.velocity(0.1) == (0.3, 0.0, 0.0)
    t.key("w", 0.5)                       # key repeat keeps it alive
    assert t.velocity(1.0) == (0.3, 0.0, 0.0)
    assert t.velocity(1.2) == (0.0, 0.0, 0.0)   # released: stops after 0.6 s


def test_turn_strafe_arrows_and_stop():
    t = Teleop(TeleopConfig())
    t.key("a", 0); assert t.velocity(0) == (0.0, 0.0, 0.6)
    t.key("e", 0); assert t.velocity(0) == (0.0, -0.3, 0.0)
    t.key("DOWN", 0); assert t.velocity(0) == (-0.3, 0.0, 0.0)
    assert t.key("ESC", 0) == "stop" and t.velocity(0) == (0.0, 0.0, 0.0)
    t.key("W", 0); assert t.velocity(0) == (0.3, 0.0, 0.0)     # caps lock still drives
    assert t.key(" ", 0) == "stop"


def test_speed_limits_and_live_rescale():
    t = Teleop(TeleopConfig(linear_speed=0.3, max_linear=0.5, step=0.1))
    t.key("w", 0)
    for _ in range(5):
        t.key("+", 0)
    assert t.linear == 0.5 and t.velocity(0)[0] == 0.5          # capped, held command rescaled
    for _ in range(10):
        t.key("-", 0)
    assert t.linear == 0.1                                       # never below one step


def test_latch_mode():
    t = Teleop(TeleopConfig(latch=True))
    t.key("w", 0)
    assert t.velocity(100.0) == (0.3, 0.0, 0.0)


def test_double_press_confirmation():
    t = Teleop(TeleopConfig(confirm_s=1.5))
    assert t.key("t", 0.0) == "confirm_stand_sit"
    assert t.key("t", 1.0) == "stand_sit"
    assert t.key("t", 5.0) == "confirm_stand_sit"
    assert t.key("t", 7.0) == "confirm_stand_sit"                # too slow: starts over
    assert t.key("r", 0) == "confirm_reset" and t.key("r", 0.2) == "reset"
    assert t.key("v", 0) == "vlm_front" and t.key("b", 0) == "vlm_down" and t.key("n", 0) == "mode_auto"


# ---------- keyboard parsing ----------
def test_parse_keys():
    assert parse_keys("wwa") == ["w", "w", "a"]
    assert parse_keys("\x1b[A\x1b[D") == ["UP", "LEFT"]
    assert parse_keys("\x1bOB") == ["DOWN"]
    assert parse_keys("\x1b") == ["ESC"]
    assert parse_keys("\x1b[1;5Cv") == ["RIGHT", "v"]            # ctrl+arrow, then v
    assert parse_keys("x\r\n") == ["x"]


# ---------- broker queue ----------
def test_queue_priority_replace_expire():
    q = RequestQueue()
    q.push(Request(id="s1", source="surface_monitor", priority=5, t_req=0.0, max_age_s=10))
    q.push(Request(id="c1", source="console", priority=10, t_req=1.0))
    replaced = q.push(Request(id="s2", source="surface_monitor", priority=5, t_req=2.0, max_age_s=10))
    assert replaced.id == "s1"                       # one waiting request per source
    nxt, expired = q.pop(3.0)
    assert nxt.id == "c1" and expired == []          # console (priority 10) first
    nxt, expired = q.pop(20.0)
    assert nxt is None and [r.id for r in expired] == ["s2"]


def test_request_from_dict_defaults():
    r = Request.from_dict({"id": "a", "source": "x"}, now=5.0)
    assert (r.camera, r.priority, r.t_req, r.snapshot_only) == ("front", 5, 5.0, False)


# ---------- gateway ----------
OK_OUT = textwrap.dedent('''\
    HTTP 200 OK
    Client elapsed: 3.0 s
    {"success": true, "group_id": "group07",
     "result": "```json\\n{\\"room_type\\": \\"kitchen\\", \\"confidence\\": 0.8}\\n```"}
''')


def test_gateway_parse():
    r = gateway.parse_client_output(OK_OUT)
    assert r["ok"] and r["parsed"]["room_type"] == "kitchen"
    bad = gateway.parse_client_output('HTTP 422\n{"success": false, "error": {"code": "invalid_group", "message": "x"}}')
    assert not bad["ok"] and bad["error"].startswith("invalid_group")
    assert gateway.parse_model_text("plain words") is None
    assert gateway.config_problems("group7", "/nope")


def test_gateway_query_with_fake_client(tmp_path):
    fake = tmp_path / "client.py"
    fake.write_text("import sys\nassert '--prompt-file' in sys.argv\n" f"print({OK_OUT!r})\n")
    img, prompt = tmp_path / "a.jpg", tmp_path / "p.txt"
    img.write_bytes(b"x")
    prompt.write_text("hi")
    r = gateway.query(sys.executable, str(fake), "http://x", "group07", str(img), str(prompt))
    assert r["ok"] and r["parsed"]["confidence"] == 0.8 and r["latency_s"] >= 0
