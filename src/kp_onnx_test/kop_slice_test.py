import numpy as np
from kp import Manager
import time
from kp_onnx_ssbo.kop_slice import SliceOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

slice_op = SliceOp(mgr)


def np_slice(data, starts, ends, axes=None, steps=None) -> np.ndarray:
    if isinstance(starts, list):
        starts = np.array(starts)
    if isinstance(ends, list):
        ends = np.array(ends)
    if isinstance(axes, list):
        axes = np.array(axes)
    if isinstance(steps, list):
        steps = np.array(steps)
    if len(starts.shape) == 0:
        starts = np.array([starts])
    if len(ends.shape) == 0:
        ends = np.array([ends])
    if axes is None:
        if steps is None:
            slices = [slice(s, e) for s, e in zip(starts, ends)]
        else:
            slices = [slice(s, e, d) for s, e, d in zip(starts, ends, steps)]
    else:
        if steps is None:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a in zip(starts, ends, axes):
                slices[a] = slice(s, e)
        else:
            slices = [slice(0, a) for a in data.shape]
            for s, e, a, d in zip(starts, ends, axes, steps):
                slices[a] = slice(s, e, d)
    try:
        return data[tuple(slices)]
    except TypeError as e:
        raise TypeError(
            f"Unable to extract slice {slices!r} for shape {data.shape!r}."
        ) from e


x = np.random.random((32, 32, 8, 32)).astype(np.float32)

print("Case 1: not axes, not steps")
starts = np.array([2, 0, 0], dtype=np.int64)
ends = np.array([30, 12, 8], dtype=np.int64)

start_time = time.time()
np_out = np_slice(x, starts=starts, ends=ends)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = slice_op.run(x, starts, ends)[0]
print(f"{slice_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 2: axes is [0, 2], not steps")
starts = np.array([0, 2, -2], dtype=np.int64)
ends = np.array([-1, 4, -1], dtype=np.int64)
axes = np.array([0, 1, -1], dtype=np.int64)

start_time = time.time()
np_out = np_slice(x, starts=starts, ends=ends, axes=axes)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = slice_op.run(x, starts, ends, axes)[0]
print(f"{slice_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 3: axes is [0,1], steps is [-1, 1000]")
starts = [2, 1]
ends = [1, 2]
axes = [0, 1]
steps = [-1, 1000]

start_time = time.time()
np_out = np_slice(x, starts=starts, ends=ends, axes=axes, steps=steps)
print("Numpy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = slice_op.run(x, starts, ends, axes, steps)[0]
print(f"{slice_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print('----')

print("Case 4: with steps")
starts = np.array([0], dtype=np.int64)
ends = np.array([30], dtype=np.int64)
axes = np.array([0], dtype=np.int64)
steps = np.array([2], dtype=np.int64)

np_out = np_slice(x, starts=starts, ends=ends, axes=axes, steps=steps)
kp_out = slice_op.run(x, starts, ends, axes, steps)[0]
print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out))

print("Case 5: empty slice")
starts = np.array([5], dtype=np.int64)
ends = np.array([5], dtype=np.int64)  # start==end → 空
axes = np.array([0], dtype=np.int64)

np_out = np_slice(x, starts=starts, ends=ends, axes=axes)
kp_out = slice_op.run(x, starts, ends, axes)[0]
print("shape:", kp_out.shape)
print("Equal:", np.array_equal(np_out, kp_out))