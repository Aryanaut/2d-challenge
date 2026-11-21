import math
import os

import cv2
import numpy as np
import pytest

from lite3_terrain_classifier import slat_lines as S
from lite3_terrain_classifier.terrain_kinds import KindClassifier

IMAGES = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'images')


def rungs_image(n=4):
    img = np.full((480, 640), 120, np.uint8)
    for i in range(n):
        y = 270 + i * 50
        cv2.line(img, (60, y), (580, y), 30, 6)
    return img


def converging_image(n=4):
    img = np.full((480, 640), 120, np.uint8)
    for i in range(n):
        x_bottom = 100 + i * 140
        cv2.line(img, (x_bottom, 479), (260 + i * 40, 250), 30, 6)
    return img


def test_rungs_are_found():
    r = S.detect(rungs_image())
    assert r['rungs'] >= 3 and S.is_slats(r)


def test_converging_lines_are_found():
    r = S.detect(converging_image())
    assert r['converging'] >= 3 and S.is_slats(r)


def test_plain_floor_is_not_slats():
    rng = np.random.RandomState(0)
    img = np.clip(120 + rng.normal(0, 3, (480, 640)), 0, 255).astype(np.uint8)
    assert not S.is_slats(S.detect(img))


def test_two_lines_are_not_enough():
    assert not S.is_slats(S.detect(rungs_image(2)))


@pytest.mark.skipif(not os.path.exists(os.path.join(IMAGES, 'depth-slats1.png')), reason='no example images')
def test_example_depth_pictures():
    slats = cv2.imread(os.path.join(IMAGES, 'depth-slats1.png'), cv2.IMREAD_GRAYSCALE)
    curtain = cv2.imread(os.path.join(IMAGES, 'depth-curtain.png'), cv2.IMREAD_GRAYSCALE)
    assert S.is_slats(S.detect(S.to_gray8(slats)))
    assert not S.is_slats(S.detect(S.to_gray8(curtain)))


def test_kind_from_images_only():
    kc = KindClassifier({'terrains': [{'name': 'slats', 'type': 'hough_slats', 'sources': ['depth', 'color'],
                                       'mode': 'any'}, {'name': 'flat', 'type': 'flat'}]})
    assert kc.classify()['kind'] == 'unknown'          # no lidar cells, no images: nothing to say
    for _ in range(6):
        kc.add_image('depth', converging_image())
    out = kc.classify()
    assert out['kind'] == 'slats' and out['slats_votes']['depth'] == 1.0


def test_mode_all_needs_every_camera():
    cfg = {'terrains': [{'name': 'slats', 'type': 'hough_slats', 'sources': ['depth', 'color'], 'mode': 'all',
                         'min_lines': {'color': {'rungs': 3, 'converging': 3}}}]}
    kc = KindClassifier(cfg)
    for _ in range(6):
        kc.add_image('depth', converging_image())
        kc.add_image('color', np.full((480, 640), 120, np.uint8))     # the colour image shows no slats
    assert kc.classify()['kind'] == 'unknown'
    kc.reset()
    for _ in range(6):
        kc.add_image('depth', converging_image())
        kc.add_image('color', rungs_image())
    assert kc.classify()['kind'] == 'slats'


def test_orientation_of_rungs_and_converging_lines():
    r = S.detect(rungs_image())
    assert r['orientation'] == 'horizontal' and abs(r['angle_deg']) < 5
    c = S.detect(converging_image())
    assert c['orientation'] == 'vertical' and 60 < c['angle_deg'] < 120
    assert S.detect(np.full((480, 640), 120, np.uint8))['orientation'] == 'none'


def test_kind_reports_the_orientation():
    cfg = {'terrains': [{'name': 'slats', 'type': 'hough_slats', 'sources': ['depth']}]}
    kc = KindClassifier(cfg)
    for _ in range(6):
        kc.add_image('depth', rungs_image())
    out = kc.classify()
    assert out['slats_orientation'] == 'horizontal' and abs(out['slats_angle_deg']) < 5


def fan_image(n=5):
    """Lines at mixed angles (like the edges of ramps and wedges): many lines, but not parallel."""
    img = np.full((480, 640), 120, np.uint8)
    for i, ang in enumerate((-18, 4, -9, 16, 0)[:n]):
        y = 280 + i * 40
        dy = int(math.tan(math.radians(ang)) * 250)
        cv2.line(img, (320 - 250, y - dy), (320 + 250, y + dy), 30, 6)
    return img


def test_lines_at_mixed_angles_are_not_slats():
    r = S.detect(fan_image())
    assert r['rungs_raw'] >= 3 and r['rungs'] == 0 and not S.is_slats(r, 3)


def _example(name):
    path = os.path.join(IMAGES, name)
    if not os.path.exists(path):
        pytest.skip('no example images')
    return S.detect(S.to_gray8(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))


@pytest.mark.parametrize('name', ['uneven-depth.png', 'uneven-color.png', 'uneven2-depth.png', 'uneven2-color.png', 'color-curtain.png',
                                  'depth-curtain.png'])
def test_uneven_and_curtain_pictures_are_not_slats(name):
    assert not S.is_slats(_example(name), 3)


@pytest.mark.parametrize('name', ['depth-slats1.png', 'f661e00f-4f02-41e1-b607-472d5ba1a935.jpeg',
                                  '58381836-9594-43fc-bd69-fe286b617f5d.jpeg'])
def test_slat_pictures_are_slats(name):
    assert S.is_slats(_example(name), 4)


def short_rungs_image():
    """Parallel but SHORT horizontal edges, staggered like the tops of a field of blocks."""
    img = np.full((480, 640), 120, np.uint8)
    for i in range(6):
        x0 = 60 + (i % 3) * 170
        cv2.line(img, (x0, 270 + i * 30), (x0 + 110, 270 + i * 30), 30, 6)
    return img


def test_short_parallel_edges_are_not_slats():
    r = S.detect(short_rungs_image())
    assert r['rungs'] == 0 and not S.is_slats(r, 3)


def test_colour_alone_cannot_name_slats_when_depth_is_required():
    kc = KindClassifier({'terrains': [{'name': 'slats', 'type': 'hough_slats', 'sources': ['depth', 'color'],
                                       'mode': 'any', 'require': ['depth']}]})
    for _ in range(6):
        kc.add_image('color', rungs_image())             # colour sees perfect rungs ...
    assert kc.classify()['kind'] == 'unknown'            # ... but depth has not confirmed
    for _ in range(6):
        kc.add_image('depth', np.full((480, 640), 120, np.uint8))        # depth sees a plain floor
    assert kc.classify()['kind'] == 'unknown'
    for _ in range(15):
        kc.add_image('depth', converging_image())        # the window (15) now holds only converging frames
    assert kc.classify()['kind'] == 'slats'


BAG = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bags', 'data', 'terrains')


@pytest.mark.skipif(not os.path.exists(os.path.join(BAG, 'terrains_0.db3')), reason='no recorded bag')
def test_recorded_run_cubes_slats_stairs():
    """bags/data/terrains: cubes (0-6 s), slats (8-12 s), metal stairs (17.5-19 s). Needs rclpy to read the bag."""
    pytest.importorskip('rclpy')
    import sqlite3

    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    db = sqlite3.connect('file:' + os.path.join(BAG, 'terrains_0.db3') + '?mode=ro', uri=True)
    types, ids = dict(db.execute('select name,type from topics')), dict(db.execute('select name,id from topics'))
    t0 = db.execute('select min(timestamp) from messages').fetchone()[0]
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'config', 'terrain_kinds.yaml')))
    windows = {'cubes': (1.0, 5.0), 'slats': (8.0, 11.5), 'stairs': (17.5, 19.0)}
    kcs = {k: KindClassifier(cfg) for k in windows}
    for topic, cam in (('/camera/depth/image_raw', 'depth'), ('/camera/color/image_raw/compressed', 'color')):
        for ts, data in db.execute('select timestamp,data from messages where topic_id=? order by timestamp', (ids[topic],)):
            t = (ts - t0) / 1e9
            for name, (a, b) in windows.items():
                if a <= t <= b and len(kcs[name].images.get(cam, [])) < 15:
                    m = deserialize_message(data, get_message(types[topic]))
                    if cam == 'depth':
                        gray = S.to_gray8(np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, m.width)
                                          .astype(np.float32) * 0.001)
                    else:
                        gray = S.to_gray8(cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_GRAYSCALE))
                    kcs[name].add_image(cam, gray)
    assert kcs['slats'].classify()['kind'] == 'slats'
    assert kcs['cubes'].classify()['kind'] != 'slats'
    assert kcs['stairs'].classify()['kind'] != 'slats'
