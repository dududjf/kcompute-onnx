from kp import Manager
import numpy as np
import time
from kp_onnx.kop_reducemean import ReduceMeanOp

# Device
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

reduce_mean_op = ReduceMeanOp(mgr)
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


def np_mean(data, axes=None, keepdims=True, noop_with_empty_axes=False):
    if axes is None and noop_with_empty_axes:
        return data
    axes = handle_axes(axes)
    try:
        res = np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype)  # type: ignore
        if not keepdims and not isinstance(res, np.ndarray):
            # The runtime must return a numpy array of a single float.
            res = np.array(res)
    except TypeError as e:
        raise TypeError(
            f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
        ) from e
    return res


# -------- Case 1 --------
print("Case 1 for keepdims and not noop_with_empty_axes: axis is None")
start_time = time.time()
np_out = np_mean(x, keepdims=True, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_mean_op.keepdims = True
reduce_mean_op.noop_with_empty_axes = False
kp_out = reduce_mean_op.run(x)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2 --------
print("Case 2 for not keepdims and not noop_with_empty_axes: axis is None")
start_time = time.time()
np_out = np_mean(x, keepdims=False, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_mean_op.keepdims = False
reduce_mean_op.noop_with_empty_axes = False
kp_out = reduce_mean_op.run(x)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3 --------
print("Case 3 for keepdims and noop_with_empty_axes: axes is None")
start_time = time.time()
np_out = np_mean(x, keepdims=True, noop_with_empty_axes=True)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_mean_op.keepdims = True
reduce_mean_op.noop_with_empty_axes = True
kp_out = reduce_mean_op.run(x, None)[0]
print(f"{reduce_mean_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4 --------
print("Case 4 for not keepdims and noop_with_empty_axes: axes is None")
start_time = time.time()
np_out = np_mean(x, keepdims=False, noop_with_empty_axes=True)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_mean_op.keepdims = False
reduce_mean_op.noop_with_empty_axes = True
kp_out = reduce_mean_op.run(x, None)[0]
print(f"{reduce_mean_op}:", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5 --------
print("Case 5 for keepdims and not noop_with_empty_axes: axes is [0,2]")
axes = np.array([0, 2], dtype=np.int32)
start_time = time.time()
np_out = np_mean(x, axes, keepdims=True, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_mean_op.keepdims = True
reduce_mean_op.noop_with_empty_axes = False
kp_out = reduce_mean_op.run(x, axes)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6 --------
print("Case 6 for not keepdims and not noop_with_empty_axes: axes is (0)")
axes = (0, )
start_time = time.time()
np_out = np_mean(x, axes, keepdims=False, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_mean_op.keepdims = False
reduce_mean_op.noop_with_empty_axes = False
kp_out = reduce_mean_op.run(x, axes)[0]
print(f"{reduce_mean_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
