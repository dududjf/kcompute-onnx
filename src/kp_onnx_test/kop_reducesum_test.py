from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_reducesum import ReduceSumOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

reduce_sum_op = ReduceSumOp(mgr)
x = np.random.random((255, 127, 63)).astype(np.float32)


def handle_axes(axes):
    if isinstance(axes, tuple):
        if len(axes) == 0:
            return None
        return axes
    if axes is None:
        return None
    if isinstance(axes, (int, tuple)):
        return axes
    if not isinstance(axes, np.ndarray):
        raise TypeError(f"axes must be an array, not {type(axes)}.")
    if len(axes.shape) == 0:
        return int(axes)
    if 0 in axes.shape:
        return None
    return tuple(axes.ravel().tolist())


def np_sum(data, axes=None, keepdims=1, noop_with_empty_axes=0):
    if axes is None and noop_with_empty_axes:
        return data
    keepdims = keepdims != 0  # type: ignore
    axes = handle_axes(axes)
    try:
        res = np.sum(data, axis=axes, keepdims=keepdims, dtype=data.dtype)  # ← 使用 np.sum
        if not keepdims and not isinstance(res, np.ndarray):
            res = np.array(res)
    except TypeError as e:
        raise TypeError(
            f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
        ) from e
    return res


# -------- Case 1 --------
print("Case 1 for keepdims and not noop_with_empty_axes: axis is None")
start_time = time.time()
np_out = np_sum(x, keepdims=True, noop_with_empty_axes=False)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = True
reduce_sum_op.noop_with_empty_axes = False
kp_out = reduce_sum_op.run(x)[0]
print(f"{reduce_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2 --------
print("Case 2 for not keepdims and not noop_with_empty_axes: axis is None")
start_time = time.time()
np_out = np_sum(x, keepdims=False, noop_with_empty_axes=False)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = False
reduce_sum_op.noop_with_empty_axes = False
kp_out = reduce_sum_op.run(x)[0]
print(f"{reduce_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3 --------
print("Case 3 for keepdims and noop_with_empty_axes: axes is None")
start_time = time.time()
np_out = np_sum(x, keepdims=True, noop_with_empty_axes=True)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = True
reduce_sum_op.noop_with_empty_axes = True
kp_out = reduce_sum_op.run(x, None)[0]
print(f"{reduce_sum_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4 --------
print("Case 4 for not keepdims and noop_with_empty_axes: axes is None")
start_time = time.time()
np_out = np_sum(x, keepdims=False, noop_with_empty_axes=True)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = False
reduce_sum_op.noop_with_empty_axes = True
kp_out = reduce_sum_op.run(x, None)[0]
print(f"{reduce_sum_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5 --------
print("Case 5 for keepdims and not noop_with_empty_axes: axes is [0,2]")
axes = np.array([1, 2], dtype=np.int32)
start_time = time.time()
np_out = np_sum(x, axes, keepdims=True, noop_with_empty_axes=False)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = True
reduce_sum_op.noop_with_empty_axes = False
kp_out = reduce_sum_op.run(x, axes)[0]
print(f"{reduce_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6 --------
print("Case 6 for not keepdims and not noop_with_empty_axes: axes is (0)")
axes = (1, )
start_time = time.time()
np_out = np_sum(x, axes, keepdims=False, noop_with_empty_axes=False)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = False
reduce_sum_op.noop_with_empty_axes = False
kp_out = reduce_sum_op.run(x, axes)[0]
print(f"{reduce_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))

# -------- Case 7 --------
# 使用负数 axis
print("Case 7 for keepdims and not noop_with_empty_axes: axes contains negative value [-1]")
axes = np.array([-1], dtype=np.int32)   # -1 等价于最后一个维度（dim=2）
start_time = time.time()
np_out = np_sum(x, axes, keepdims=True, noop_with_empty_axes=False)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = True
reduce_sum_op.noop_with_empty_axes = False
kp_out = reduce_sum_op.run(x, axes)[0]
print(f"{reduce_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 8 --------
# 使用多个负数 axis
print("Case 8 for not keepdims and not noop_with_empty_axes: axes contains negative values [-2, -1]")
axes = np.array([-2, -1], dtype=np.int32)   # 等价于 [1, 2]
start_time = time.time()
np_out = np_sum(x, axes, keepdims=False, noop_with_empty_axes=False)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_sum_op.keepdims = False
reduce_sum_op.noop_with_empty_axes = False
kp_out = reduce_sum_op.run(x, axes)[0]
print(f"{reduce_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
