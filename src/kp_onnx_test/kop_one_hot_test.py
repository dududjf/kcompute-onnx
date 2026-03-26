from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_one_hot import OneHotOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

one_hot_op = OneHotOp(mgr)


def _one_hot(indices, depth, axis=-1, dtype=np.float32):
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += rank + 1
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    new_shape = (1,) * len(ls) + depth_range.shape + (1,) * len(rs)
    targets = np.reshape(depth_range, new_shape)
    values = np.reshape(np.mod(values, depth), (*ls, 1, *rs))
    return np.asarray(targets == values, dtype=dtype)


def np_one_hot(indices, depth, values, axis=-1):
    off_value, on_value = values
    y = _one_hot(indices, depth, axis=axis, dtype=values.dtype)
    y = y * (on_value - off_value) + off_value
    return y


# Case 1: 1D输入，默认轴（-1）
print('Case 1: 1D input, default axis (-1)')
indices = np.array([0, 1, -7, 1, 0], dtype=np.int32)
depth = 3
values = np.array([0.0, 1.0], dtype=np.float32)

start_time = time.time()
np_out = np_one_hot(indices, depth, values)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = one_hot_op.run(indices, depth, values)[0]
print(f"{one_hot_op}: ", time.time() - start_time, "seconds")

print('Max error:', np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 2: 2D输入，轴=2
print('Case 2: 2D input, axis=2')
indices = np.array([[0, 1], [2, 1]], dtype=np.int32)
depth = 3
values = np.array([0.0, 1.0], dtype=np.float32)
axis = 2

start_time = time.time()
np_out = np_one_hot(indices, depth, values, axis)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
one_hot_op.axis = 2
kp_out = one_hot_op.run(indices, depth, values)[0]
print(f"{one_hot_op}: ", time.time() - start_time, "seconds")

print('Max error:', np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

# Case 3: 3D输入, depth=0
print('Case 3: 3D input, depth=0')
indices = np.array([[[0, 1], [2, 1]], [[1, 0], [2, 2]]], dtype=np.int32)
depth = 0
values = np.array([0.0, 1.0], dtype=np.float32)
axis = 0

start_time = time.time()
np_out = np_one_hot(indices, depth, values, axis)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
one_hot_op.axis = 0
kp_out = one_hot_op.run(indices, depth, values)[0]
print(f"{one_hot_op}: ", time.time() - start_time, "seconds")

print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
