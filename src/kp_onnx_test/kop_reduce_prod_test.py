from kp import Manager
import numpy as np
import time
from kp_onnx.kop_reduce_prod import ReduceProdOp


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

reduce_prod_op = ReduceProdOp(mgr)
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

def np_prod(data, axes=None, keepdims=True, noop_with_empty_axes=False):
    if axes is None and noop_with_empty_axes:
        return data
    axes = handle_axes(axes)
    try:
        res = np.prod(data, axis=axes, keepdims=keepdims, dtype=data.dtype)
        if not keepdims and not isinstance(res, np.ndarray):
            res = np.array(res)
    except TypeError as e:
        raise TypeError(
            f"Unable to reduce shape {data.shape!r} with axes={axes!r} and keepdims={keepdims}."
        ) from e
    return res


print("Case 1: keepdims=True, noop_with_empty_axes=False, axes=None")
start_time = time.time()
np_out = np_prod(x, keepdims=True, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_prod_op.keepdims = True
reduce_prod_op.noop_with_empty_axes = False
kp_out = reduce_prod_op.run(x)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


print("Case 2: keepdims=False, noop_with_empty_axes=False, axes=None")
start_time = time.time()
np_out = np_prod(x, keepdims=False, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_prod_op.keepdims = False
reduce_prod_op.noop_with_empty_axes = False
kp_out = reduce_prod_op.run(x)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


print("Case 3: keepdims=True, noop_with_empty_axes=True, axes=None")
start_time = time.time()
np_out = np_prod(x, keepdims=True, noop_with_empty_axes=True)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_prod_op.keepdims = True
reduce_prod_op.noop_with_empty_axes = True
kp_out = reduce_prod_op.run(x, None)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


print("Case 4: keepdims=False, noop_with_empty_axes=True, axes=None")
start_time = time.time()
np_out = np_prod(x, keepdims=False, noop_with_empty_axes=True)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_prod_op.keepdims = False
reduce_prod_op.noop_with_empty_axes = True
kp_out = reduce_prod_op.run(x, None)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


print("Case 5: keepdims=True, axes=[0,2]")
axes = np.array([0, 2], dtype=np.int32)
start_time = time.time()
np_out = np_prod(x, axes, keepdims=True, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_prod_op.keepdims = True
reduce_prod_op.noop_with_empty_axes = False
kp_out = reduce_prod_op.run(x, axes)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")


print("Case 6: keepdims=False, axes=(0,)")
axes = (0,)
start_time = time.time()
np_out = np_prod(x, axes, keepdims=False, noop_with_empty_axes=False)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
reduce_prod_op.keepdims = False
reduce_prod_op.noop_with_empty_axes = False
kp_out = reduce_prod_op.run(x, axes)[0]
print(f"{reduce_prod_op}: ", time.time() - start_time, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
