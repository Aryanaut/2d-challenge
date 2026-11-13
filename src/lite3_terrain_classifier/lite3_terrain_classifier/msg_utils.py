"""Turn sensor_msgs PointCloud2 / Image into numpy arrays without cv_bridge."""
import numpy as np

# sensor_msgs/PointField datatypes
_FIELD_DTYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def _buffer(data):
    try:
        return memoryview(data)      # array.array / bytes: zero copy
    except TypeError:
        return bytes(data)          # plain list


def cloud_to_xyz(msg) -> np.ndarray:
    """PointCloud2 -> (N, 3) float64 with NaN / inf points removed."""
    fields = {f.name: f for f in msg.fields}
    missing = [k for k in ('x', 'y', 'z') if k not in fields]
    if missing:
        raise ValueError(f'PointCloud2 has no {missing} field(s); fields are {list(fields)}')
    order = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': [order + _FIELD_DTYPES[fields[k].datatype] for k in ('x', 'y', 'z')],
        'offsets': [fields[k].offset for k in ('x', 'y', 'z')],
        'itemsize': msg.point_step,
    })
    n_cols = msg.width
    buf = _buffer(msg.data)
    if msg.row_step == n_cols * msg.point_step or msg.height <= 1:
        arr = np.frombuffer(buf, dtype=dtype, count=msg.width * msg.height)
    else:
        rows = [np.frombuffer(buf, dtype=dtype, count=n_cols, offset=r * msg.row_step)
                for r in range(msg.height)]
        arr = np.concatenate(rows)
    xyz = np.column_stack((arr['x'], arr['y'], arr['z'])).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def depth_to_metres(msg, mm_scale: float = 0.001) -> np.ndarray:
    """sensor_msgs/Image depth (16UC1 in mm or 32FC1 in m) -> (H, W) float32 metres, NaN = no data."""
    enc = msg.encoding.lower()
    if enc in ('16uc1', 'mono16'):
        base = np.dtype('u2')
    elif enc == '32fc1':
        base = np.dtype('f4')
    else:
        raise ValueError(f'unsupported depth encoding {msg.encoding!r} (need 16UC1 or 32FC1)')
    dtype = base.newbyteorder('>' if msg.is_bigendian else '<')
    buf = _buffer(msg.data)
    per_row = msg.step // dtype.itemsize
    img = np.frombuffer(buf, dtype=dtype, count=per_row * msg.height).reshape(msg.height, per_row)
    img = img[:, :msg.width]
    if base.kind == 'u':
        out = img.astype(np.float32) * np.float32(mm_scale)
        out[img == 0] = np.nan
    else:
        out = img.astype(np.float32)
        out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


def xyzi_cloud(header, points: np.ndarray, intensity: np.ndarray, cloud_cls, field_cls):
    """Build a PointCloud2 with x, y, z, intensity float32 fields."""
    data = np.column_stack((points, intensity)).astype('<f4')
    msg = cloud_cls()
    msg.header = header
    msg.height = 1
    msg.width = len(data)
    msg.fields = [field_cls(name=n, offset=4 * i, datatype=7, count=1)
                  for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(data)
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg
