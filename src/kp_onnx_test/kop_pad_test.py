from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_pad import PadOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

pad_op = PadOp(mgr)


def np_pad(data, pads, constant_value=None, axes=None, mode='constant'):
    if constant_value is None:
        constant_value = 0
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)
    if num_axes * 2 != len(pads):
        raise RuntimeError(
            "The number of elements in raw_pads should be 2 times the number of axes"
        )

    pad_width = [(0, 0)] * input_rank
    for i, axis in enumerate(axes):
        pad_begin = pads[i]
        pad_end = pads[num_axes + i]
        pad_width[axis] = (pad_begin, pad_end)

    if mode == "constant":
        return np.pad(
            data, pad_width=pad_width, mode=mode, constant_values=constant_value
        ).astype(data.dtype)
    return np.pad(data, pad_width=pad_width, mode=mode).astype(data.dtype)


# -------- Case 1: 2D constant padding, axes: None --------
print("Case 1: 2D constant padding, axes: None")
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)
pads = np.array([1, 2, 1, 2], dtype=np.int64)  # top, left, bottom, right
constant_value = np.array([0.0], dtype=np.float32)
mode = 'constant'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: 2D edge padding --------
print("Case 2: 2D edge padding")
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]], dtype=np.float32)
pads = np.array([1, 1, 1, 1], dtype=np.int64)
mode = 'edge'

start_time = time.time()
np_out = np_pad(x, pads, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: 2D reflect padding --------
print("Case 3: 2D reflect padding")
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]], dtype=np.float32)
pads = np.array([1, 1, 1, 1], dtype=np.int64)
mode = 'reflect'

start_time = time.time()
np_out = np_pad(x, pads, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: 2D wrap padding --------
print("Case 4: 2D wrap padding")
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]], dtype=np.float32)
pads = np.array([1, 1, 1, 1], dtype=np.int64)
mode = 'wrap'

start_time = time.time()
np_out = np_pad(x, pads, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5: 1D constant padding --------
print("Case 5: 1D constant padding")
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
pads = np.array([3, 3], dtype=np.int64)  # left, right
constant_value = np.array([-1.0], dtype=np.int64)
mode = 'constant'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6: 1D edge padding --------
print("Case 6: 1D edge padding")
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
pads = np.array([3, 3], dtype=np.int64)  # left, right
mode = 'edge'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 7: 1D reflect padding --------
print("Case 7: 1D reflect padding")
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
pads = np.array([3, 3], dtype=np.int64)  # left, right
mode = 'reflect'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 8: 1D wrap padding --------
print("Case 8: 1D wrap padding")
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
pads = np.array([3, 3], dtype=np.int64)  # left, right
mode = 'wrap'

start_time = time.time()
np_out = np_pad(x, pads, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 9: 3D with axes parameter, mode: constant --------
print("Case 9: 3D with axes parameter (pad only axis 1,2), mode: constant")
x = np.array([[[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]],

              [[7.0, 8.0, 9.0],
               [10.0, 11.0, 12.0]],

              [[16.0, 17.0, 18.0],
               [13.0, 14.0, 15.0]]], dtype=np.float32)
pads = np.array([1, 2, 1, 2], dtype=np.int64)  # left, right for axis 1 only
axes = np.array([0, 2], dtype=np.int64)
constant_value = np.array([888.0], dtype=np.float32)
mode = 'constant'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, axes=axes, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value, axes)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 10: 3D with axes parameter, mode: edge --------
print("Case 10: 3D with axes parameter, mode: edge")
x = np.array([[[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]],

              [[7.0, 8.0, 9.0],
               [10.0, 11.0, 12.0]],

              [[16.0, 17.0, 18.0],
               [13.0, 14.0, 15.0]]], dtype=np.float32)
pads = np.array([1, 1, 3, 3], dtype=np.int64)  # all dimensions
axes = np.array([0, 2], dtype=np.int64)
constant_value = np.array([-99.0], dtype=np.float32)
mode = 'edge'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, axes=axes, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value, axes)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 11: 3D with axes parameter, mode: reflect --------
print("Case 11: 3D with axes parameter, mode: reflect")
x = np.array([[[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]],

              [[7.0, 8.0, 9.0],
               [10.0, 11.0, 12.0]],

              [[16.0, 17.0, 18.0],
               [13.0, 14.0, 15.0]]], dtype=np.float32)
pads = np.array([10, 20, 30, 40], dtype=np.int64)
axes = np.array([0, 2], dtype=np.int64)
constant_value = np.array([0.0], dtype=np.float32)
mode = 'reflect'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, axes=axes, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value, axes)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 12: 3D with axes parameter, mode: wrap --------
print("Case 12: 3D with axes parameter, mode: wrap")
x = np.array([[[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]],

              [[7.0, 8.0, 9.0],
               [10.0, 11.0, 12.0]],

              [[16.0, 17.0, 18.0],
               [13.0, 14.0, 15.0]]], dtype=np.float32)
pads = np.array([10, 20, 30, 40], dtype=np.int64)
axes = np.array([0, 2], dtype=np.int64)
constant_value = np.array([0.0], dtype=np.float32)
mode = 'wrap'

start_time = time.time()
np_out = np_pad(x, pads, constant_value=constant_value, axes=axes, mode=mode)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
pad_op.set_mode(mode)
kp_out = pad_op.run(x, pads, constant_value, axes)[0]
print(f"{pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
