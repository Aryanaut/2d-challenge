"""Image helpers shared by the nodes (no cv_bridge needed)."""
import math

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage


def image_msg_to_bgr(msg):
    """sensor_msgs/Image or CompressedImage -> BGR numpy array (no cv_bridge needed)."""
    if isinstance(msg, CompressedImage):
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'could not decode compressed image ({msg.format})')
        return img
    enc = msg.encoding.lower()
    raw = np.frombuffer(bytes(msg.data), np.uint8)
    channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1, '8uc1': 1, '8uc3': 3,
                'yuv422': 2, 'uyvy': 2, 'yuv422_yuy2': 2, 'yuyv': 2}.get(enc)
    if channels is None:
        raise ValueError(f'unsupported image encoding {msg.encoding}')
    rows = raw.reshape(msg.height, msg.step)[:, :msg.width * channels]
    img = rows.reshape(msg.height, msg.width, channels) if channels > 1 else rows.reshape(msg.height, msg.width)
    conversions = {'rgb8': cv2.COLOR_RGB2BGR, 'bgra8': cv2.COLOR_BGRA2BGR, 'rgba8': cv2.COLOR_RGBA2BGR,
                   'mono8': cv2.COLOR_GRAY2BGR, '8uc1': cv2.COLOR_GRAY2BGR,
                   'yuv422': cv2.COLOR_YUV2BGR_UYVY, 'uyvy': cv2.COLOR_YUV2BGR_UYVY,
                   'yuv422_yuy2': cv2.COLOR_YUV2BGR_YUY2, 'yuyv': cv2.COLOR_YUV2BGR_YUY2}
    return cv2.cvtColor(img, conversions[enc]) if enc in conversions else img.copy()


def encode_jpeg(bgr, max_width, quality):
    if max_width > 0 and bgr.shape[1] > max_width:
        scale = max_width / bgr.shape[1]
        bgr = cv2.resize(bgr, (max_width, int(round(bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return buf.tobytes()


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))
