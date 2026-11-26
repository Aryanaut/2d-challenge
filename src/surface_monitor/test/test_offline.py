"""Offline tests (no ROS, no network): python3 -m pytest test/ -q"""

import os
import stat
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from surface_monitor.fusion import FusionConfig, SurfaceFusion  # noqa: E402
from surface_monitor.vlm_client import (GatewayClient, parse_client_output,  # noqa: E402
                                        parse_surface_result)

OK_OUTPUT = textwrap.dedent('''\
    HTTP 200 OK
    Client elapsed: 3.1 s
    {
      "success": true,
      "group_id": "group07",
      "request_id": "abc",
      "result": "{\\"surface\\":\\"metal\\",\\"confidence\\":0.92,\\"evidence\\":\\"diamond tread plate\\"}",
      "processing_time_s": 2.9
    }
''')

BAD_GROUP_OUTPUT = textwrap.dedent('''\
    HTTP 422 Unprocessable Entity
    Client elapsed: 0.266 s
    {
      "success": false,
      "group_id": null,
      "request_id": "ec05",
      "error": {"code": "invalid_group", "message": "Invalid group ID or code."}
    }
    Server rejected or could not complete the request.
''')


def test_parse_success():
    r = parse_client_output(OK_OUTPUT)
    assert r.ok and r.surface == "metal" and abs(r.confidence - 0.92) < 1e-6


def test_parse_error():
    r = parse_client_output(BAD_GROUP_OUTPUT)
    assert not r.ok and "invalid_group" in r.error


def test_parse_fenced_and_free_text():
    assert parse_surface_result('```json\n{"surface": "Wood", "confidence": 0.8}\n```')[0] == "wood"
    assert parse_surface_result("It looks like grey vinyl floor")[0] == "other"
    assert parse_surface_result({"surface": "aluminium plate", "confidence": "0.7"})[:2] == ("metal", 0.7)


def test_gateway_subprocess(tmp_path):
    fake = tmp_path / "client.py"
    fake.write_text(
        "import sys, os\n"
        "a = sys.argv\n"
        "assert a[a.index('--group') + 1] == 'group07'\n"
        "assert os.path.isfile(a[a.index('--image') + 1])\n"
        f"print({OK_OUTPUT!r})\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x")
    c = GatewayClient(sys.executable, str(fake), "http://x", "group07", str(prompt))
    assert c.config_problems() == []
    r = c.query(np.zeros((1080, 1920, 3), np.uint8))
    assert r.ok and r.surface == "metal"
    assert GatewayClient(sys.executable, str(fake), "http://x", "group7", str(prompt)).config_problems()


def _run(f, t0, t1, label, dt=0.1):
    t = t0
    while t < t1:
        f.add_local(t, label)
        f.step(t)
        t += dt
    return t


def test_full_run_hybrid_with_vlm_confirmation():
    f = SurfaceFusion(FusionConfig())
    t = _run(f, 0, 2, "other")                 # start area on lab floor: must NOT complete
    assert f.state == "other" and not f.course_complete
    t = _run(f, t, 10, "wood")
    assert f.state == "wood"
    t = _run(f, t, 14, "metal")
    assert f.state == "metal"
    t = _run(f, t, 20, "wood")
    t_exit = t
    t = _run(f, t, t + 3, "other")             # left the course
    assert f.state == "other" and not f.course_complete   # waits for the VLM
    f.add_vlm(t, t_exit + 1.5, True, "other", 0.9)
    t = _run(f, t, t + 0.5, "other")
    assert f.course_complete and "VLM" in f.complete_reason


def test_vlm_veto_then_fallback():
    f = SurfaceFusion(FusionConfig())
    t = _run(f, 0, 10, "wood")
    t_exit = t
    t = _run(f, t, t + 3, "other")
    f.add_vlm(t, t_exit + 1.0, True, "wood", 0.9)   # VLM says still on the course
    t = _run(f, t, t + 1, "other")
    assert not f.course_complete
    t = _run(f, t, t + 16, "other")                 # gateway then goes silent
    assert f.course_complete and "local only" in f.complete_reason


def test_brief_glitch_does_not_complete():
    f = SurfaceFusion(FusionConfig(mode="local_only"))
    t = _run(f, 0, 10, "wood")
    t = _run(f, t, t + 0.4, "other")    # a few bad frames
    t = _run(f, t, t + 2, "wood")
    assert not f.course_complete and f.state == "wood"


def test_camera_timeout_unknown():
    f = SurfaceFusion(FusionConfig())
    _run(f, 0, 5, "wood")
    assert f.step(7.0) == "unknown"


def test_vlm_scheduling():
    f = SurfaceFusion(FusionConfig())
    f.add_local(0, "wood")
    f.step(0)
    assert f.should_query_vlm(0)
    f.mark_vlm_requested(0)
    assert not f.should_query_vlm(1)       # in flight
    f.add_vlm(4, 0, True, "wood", 0.9)
    assert not f.should_query_vlm(4) or f.urgent  # heartbeat not due unless something changed
    assert f.should_query_vlm(8.5)         # heartbeat


def _synthetic_surface(label, rng):
    """Crude stand-ins: wood = warm blotchy flakes, metal = grey diamond pattern + highlights,
    other = smooth grey with fine speckle."""
    import cv2
    h, w = 480, 640
    if label == "wood":
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (70, 140, 190)
        for _ in range(250):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            c = tuple(int(v) for v in rng.integers(40, 90, 1)) + (int(rng.integers(110, 170)), int(rng.integers(160, 220)))
            cv2.ellipse(img, (x, y), (int(rng.integers(10, 40)), int(rng.integers(3, 8))),
                        float(rng.uniform(0, 180)), 0, 360, (c[0], c[1], c[2]), -1)
    elif label == "metal":
        yy, xx = np.mgrid[0:h, 0:w]
        pat = (np.sin((xx + yy) / 6.0) > 0.6) | (np.sin((xx - yy) / 6.0) > 0.6)
        g = np.where(pat, 220, 120).astype(np.uint8) + rng.integers(0, 20, (h, w)).astype(np.uint8)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    else:
        g = (150 + rng.normal(0, 4, (h, w))).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return img


def test_train_and_load_model(tmp_path):
    import cv2
    from surface_monitor import train
    from surface_monitor.model_classifier import ModelClassifier
    rng = np.random.default_rng(0)
    for label in ("wood", "metal", "other"):
        for sess in ("run1", "run2"):
            d = tmp_path / "data" / label / sess
            d.mkdir(parents=True)
            for i in range(6):
                cv2.imwrite(str(d / f"{i}.jpg"), _synthetic_surface(label, rng))
    out = tmp_path / "model.joblib"
    train.main([str(tmp_path / "data"), "--out", str(out), "--augment", "1"])
    clf = ModelClassifier(str(out), min_conf=0.0)
    for label in ("wood", "metal", "other"):
        pred, conf, probs = clf.predict(_synthetic_surface(label, np.random.default_rng(99)))
        assert pred == label, (label, probs)
