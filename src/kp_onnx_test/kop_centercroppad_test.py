from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_centercroppad import CenterCropPadOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

center_crop_pad_op = CenterCropPadOp(mgr)


def np_center_crop_pad(input_data, shape, axes=None):
    """Reference NumPy implementation of CenterCropPad."""
    input_rank = len(input_data.shape)
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [a if a >= 0 else a + input_rank for a in axes]

    pad_slices = [slice(0, s) for s in input_data.shape]
    crop_slices = [slice(0, s) for s in input_data.shape]
    new_shape = list(input_data.shape)

    for a, sh in zip(axes, shape):
        dim = input_data.shape[a]
        if sh == dim:
            pass
        elif sh < dim:
            # Center crop
            new_shape[a] = sh
            d = dim - sh
            if d % 2 == 0:
                d //= 2
                crop_slices[a] = slice(d, dim - d)
            else:
                d //= 2
                crop_slices[a] = slice(d, dim - d - 1)
        else:
            # Center pad
            new_shape[a] = sh
            d = sh - dim
            if d % 2 == 0:
                d //= 2
                pad_slices[a] = slice(d, sh - d)
            else:
                d //= 2
                pad_slices[a] = slice(d, sh - d - 1)

    res = np.zeros(tuple(new_shape), dtype=input_data.dtype)
    cropped = input_data[tuple(crop_slices)]
    res[tuple(pad_slices)] = cropped
    return res



# -------- Case 1: 2D crop only (smaller target) --------
print("Case 1: 2D center crop")
x = np.random.randn(1, 1, 6, 8).astype(np.float32)
shape = np.array([1, 1, 4, 4], dtype=np.int64)

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 2: 2D pad only (larger target) --------
print("Case 2: 2D center pad")
x = np.random.randn(1, 1, 3, 3).astype(np.float32)
shape = np.array([1, 1, 7, 7], dtype=np.int64)

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 3: Mixed crop and pad --------
print("Case 3: Mixed crop and pad")
x = np.random.randn(2, 3, 8, 4).astype(np.float32)
shape = np.array([2, 3, 4, 8], dtype=np.int64)  # crop axis 2, pad axis 3

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 4: With axes parameter --------
print("Case 4: With axes parameter")
x = np.random.randn(2, 3, 8, 6).astype(np.float32)
shape_target = np.array([4, 10], dtype=np.int64)  # crop axis 2 to 4, pad axis 3 to 10

start_time = time.time()
np_out = np_center_crop_pad(x, shape_target, axes=[2, 3])
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
op4 = CenterCropPadOp(mgr, axes=[2, 3])
kp_out = op4.run(x, shape_target)[0]
print(f"{op4}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 5: 1D tensor --------
print("Case 5: 1D center crop/pad")
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
shape = np.array([5], dtype=np.int64)

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 6: 1D pad --------
print("Case 6: 1D center pad")
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
shape = np.array([7], dtype=np.int64)

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 7: No change (same shape) --------
print("Case 7: Same shape (no-op)")
x = np.random.randn(2, 3, 4).astype(np.float32)
shape = np.array([2, 3, 4], dtype=np.int64)

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 8: Odd difference (asymmetric) --------
print("Case 8: Odd difference crop")
x = np.random.randn(1, 1, 7, 7).astype(np.float32)
shape = np.array([1, 1, 4, 4], dtype=np.int64)  # diff=3, crop_begin=1, crop_end=2

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 9: Odd difference pad --------
print("Case 9: Odd difference pad")
x = np.random.randn(1, 1, 3, 3).astype(np.float32)
shape = np.array([1, 1, 6, 6], dtype=np.int64)  # diff=3, pad_begin=1, pad_end=2

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 10: Large random tensor --------
print("Case 10: Large random tensor")
x = np.random.randn(4, 16, 64, 64).astype(np.float32)
shape = np.array([4, 16, 48, 80], dtype=np.int64)

start_time = time.time()
np_out = np_center_crop_pad(x, shape)
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
kp_out = center_crop_pad_op.run(x, shape)[0]
print(f"{center_crop_pad_op}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")

# -------- Case 11: Negative axes --------
print("Case 11: Negative axes")
x = np.random.randn(2, 3, 8, 6).astype(np.float32)
shape_target = np.array([4, 10], dtype=np.int64)

start_time = time.time()
np_out = np_center_crop_pad(x, shape_target, axes=[-2, -1])
print("NumPy:", time.time() - start_time, "seconds")

start_time = time.time()
op11 = CenterCropPadOp(mgr, axes=[-2, -1])
kp_out = op11.run(x, shape_target)[0]
print(f"{op11}: ", time.time() - start_time, "seconds")

print("Max error:", np.abs(np_out - kp_out).max())
print(np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("----")
