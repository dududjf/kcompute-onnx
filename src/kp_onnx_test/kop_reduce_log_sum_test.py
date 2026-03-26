from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_reduce_log_sum import ReduceLogSumOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

reduce_log_sum_op = ReduceLogSumOp(mgr)
x = np.random.random((255, 1024, 1024)).astype(np.float32)


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


def reduce_constant(data, const_val, axes, keepdims):
    """Special case reduction where the output value is a constant."""
    out_shape = np.sum(data, axis=axes, keepdims=keepdims).shape
    return np.full(out_shape, const_val, dtype=data.dtype)


def np_reduce_log_sum(data, axes=None, keepdims=True, noop_with_empty_axes=False):
    if axes is None and noop_with_empty_axes:
        return data
    axes = handle_axes(axes)
    if data.size == 0:
        return reduce_constant(data, -np.inf, axes, keepdims)

    res = np.sum(data, axis=axes, keepdims=keepdims)
    if len(res.shape) > 0:
        return np.log(res, out=res)
    return np.log(res)

# -------- Case 1 --------
print("Case 1 for keepdims and not noop_with_empty_axes: axis is None")
start_time = time.time()
np_out = np_reduce_log_sum(x, keepdims=True, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = True
reduce_log_sum_op.noop_with_empty_axes = False
kp_out = reduce_log_sum_op.run(x)[0]
print(f"{reduce_log_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2 --------
print("Case 2 for not keepdims and not noop_with_empty_axes: axis is None")
start_time = time.time()
np_out = np_reduce_log_sum(x, keepdims=False, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = False
reduce_log_sum_op.noop_with_empty_axes = False
kp_out = reduce_log_sum_op.run(x)[0]
print(f"{reduce_log_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3 --------
print("Case 3 for keepdims and noop_with_empty_axes: axes is None")
start_time = time.time()
np_out = np_reduce_log_sum(x, keepdims=True, noop_with_empty_axes=True)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = True
reduce_log_sum_op.noop_with_empty_axes = True
kp_out = reduce_log_sum_op.run(x, None)[0]
print(f"{reduce_log_sum_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4 --------
print("Case 4 for not keepdims and noop_with_empty_axes: axes is None")
start_time = time.time()
np_out = np_reduce_log_sum(x, keepdims=False, noop_with_empty_axes=True)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = False
reduce_log_sum_op.noop_with_empty_axes = True
kp_out = reduce_log_sum_op.run(x, None)[0]
print(f"{reduce_log_sum_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5 --------
print("Case 5 for keepdims and not noop_with_empty_axes: axes is [0,2]")
axes = np.array([0, 2], dtype=np.int32)
start_time = time.time()
np_out = np_reduce_log_sum(x, axes, keepdims=True, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = True
reduce_log_sum_op.noop_with_empty_axes = False
kp_out = reduce_log_sum_op.run(x, axes)[0]
print(f"{reduce_log_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6 --------
print("Case 6 for not keepdims and not noop_with_empty_axes: axes is (0)")
axes = (0, )
start_time = time.time()
np_out = np_reduce_log_sum(x, axes, keepdims=False, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = False
reduce_log_sum_op.noop_with_empty_axes = False
kp_out = reduce_log_sum_op.run(x, axes)[0]
print(f"{reduce_log_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 7 --------
print("Case 7 for keepdims and not noop_with_empty_axes: axes is (0), x.size is 0")
x = np.array([], dtype=np.float32)
axes = (0, )

start_time = time.time()
np_out = np_reduce_log_sum(x, axes, keepdims=True, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = True
reduce_log_sum_op.noop_with_empty_axes = False
kp_out = reduce_log_sum_op.run(x, axes)[0]
print(f"{reduce_log_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 8 --------
print("Case 8 for not keepdims and not noop_with_empty_axes: x.size is 0")
x = np.array([], dtype=np.float32)

start_time = time.time()
np_out = np_reduce_log_sum(x, keepdims=False, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_log_sum_op.keepdims = False
reduce_log_sum_op.noop_with_empty_axes = False
kp_out = reduce_log_sum_op.run(x)[0]
print(f"{reduce_log_sum_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))