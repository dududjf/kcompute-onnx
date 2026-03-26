from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_col2im import Col2imOp

device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())


def _get_indices(i, shape):
    res = np.empty((len(shape),), dtype=np.int64)
    k = len(shape) - 1
    while k > 0:
        m = i % shape[k]
        res[k] = m
        i -= m
        i /= shape[k]
        k -= 1
    res[0] = i
    return res


def _is_out(ind, shape):  # type: ignore
    for i, s in zip(ind, shape):
        if i < 0:
            return True
        if i >= s:
            return True
    return False


def _col2im_shape_check(X, output_shape, kernel_shape, dilations, pads, strides):
    n_input_plane = X.shape[0]
    kernel_size = np.prod(kernel_shape)
    if n_input_plane % kernel_size != 0:
        raise ValueError(
            f"Expected size of input's dimension 1 to be divisible by the "
            f"product of kernel_size={kernel_size}, "
            f"but got input.size(1)={n_input_plane} "
            f"and kernel_shape={kernel_shape}, X.shape={X.shape}, output_shape={output_shape}."
        )
    input_length = X.shape[1]
    n_dims = len(output_shape)
    n_blocks = []
    for i in range(n_dims):
        n_block = (
            output_shape[i]
            + pads[i, :].sum()
            - dilations[i] * (kernel_shape[i] - 1)
            - 1
        ) // strides[i] + 1
        n_blocks.append(n_block)
    block_size = np.prod(n_blocks)
    if input_length != block_size:
        raise ValueError(
            f"Given n_input_plane={n_input_plane}, X.shape={X.shape}, "
            f"output_shape={output_shape}, kernel_shape={kernel_shape}, "
            f"dilations={dilations}, pads={pads}, strides={strides}, "
            f"expected size of input's dimension 2 to match the calculated number of "
            f"sliding blocks {n_blocks} = {block_size}, "
            f"but got input.size(2)={input_length}.",
        )


def col2im_naive_implementation(data, image_shape, kernel_shape, dilations, pads, strides):
    """Naive implementation for `col2im`."""
    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    _col2im_shape_check(data, image_shape, kernel_shape, dilations, new_pads, strides)

    data_col = data
    data_im = np.zeros(image_shape, dtype=data.dtype)

    dim_col = []
    for i in range(n_dims):
        col = (
            image_shape[i]
            + new_pads[i, :].sum()
            - (dilations[i] * (kernel_shape[i] - 1) + 1)
        ) // strides[i] + 1
        dim_col.append(col)

    kernel_size = np.prod(kernel_shape)
    col_size = np.prod(dim_col)
    for c_col in range(kernel_size):
        offset = _get_indices(c_col, kernel_shape)

        for col in range(col_size):
            ind_col = _get_indices(col, dim_col)
            ind_im = []
            for i in range(n_dims):
                ind = (
                    ind_col[i] * strides[i] - new_pads[i, 0] + offset[i] * dilations[i]
                )
                ind_im.append(ind)

            if not _is_out(ind_im, data_im.shape):
                data_im[tuple(ind_im)] += data_col[c_col, col]

    return data_im


def onnx_col2im(data, image_shape, block_shape, dilations=None, pads=None, strides=None):
    if dilations is None:
        dilations = [1 for s in image_shape]
    if pads is None:
        pads = [0 for s in image_shape] * 2
    if strides is None:
        strides = [1 for s in image_shape]

    bl = np.prod(block_shape)
    C = data.shape[1] // bl
    data = data.reshape(data.shape[:1] + (C,) + (bl,) + data.shape[2:])

    ks = tuple(block_shape)
    res = None
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            out = col2im_naive_implementation(
                data[n, c, ...], image_shape, ks, dilations, pads, strides
            )
            if res is None:
                new_shape = data.shape[:2] + out.shape
                res = np.empty(new_shape, dtype=data.dtype)
            res[n, c, ...] = out
    return res


# ==================== Test Cases ====================

# ---------------- Case 1: Basic 2D, stride=1, no padding ----------------
print("Case 1: Basic 2D, stride=1, defult padding")
N, C, H, W = 1, 1, 4, 4
kernel_h, kernel_w = 2, 2
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 0, 0

# Calculate L (number of blocks)
height_col = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
width_col = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

# Input data: [N, C * block_size, L]
data = np.arange(1, N * C * block_size * L + 1, dtype=np.float32).reshape(N, C * block_size, L)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape,
                     dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr, dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

# ---------------- Case 2: With stride=2 ----------------
print("Case 2: stride=2")
N, C, H, W = 1, 1, 5, 5
kernel_h, kernel_w = 2, 2
stride_h, stride_w = 2, 2
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 0, 0

height_col = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
width_col = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

data = np.random.random((N, C * block_size, L)).astype(np.float32)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape,
                     dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr, dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

# ---------------- Case 3: With padding ----------------
print("Case 3: With padding")
N, C, H, W = 1, 1, 4, 4
kernel_h, kernel_w = 3, 3
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 1, 1

height_col = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
width_col = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

data = np.random.random((N, C * block_size, L)).astype(np.float32)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape,
                     dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr, dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

# ---------------- Case 4: With dilation ----------------
print("Case 4: With dilation")
N, C, H, W = 1, 1, 6, 6
kernel_h, kernel_w = 2, 2
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 2, 2
pad_h, pad_w = 0, 0

height_col = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
width_col = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

data = np.random.random((N, C * block_size, L)).astype(np.float32)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape,
                     dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr, dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

# ---------------- Case 5: Multiple channels ----------------
print("Case 5: Multiple channels")
N, C, H, W = 1, 3, 4, 4
kernel_h, kernel_w = 2, 2
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 0, 0

height_col = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
width_col = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

data = np.random.random((N, C * block_size, L)).astype(np.float32)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape,
                     dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr, dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

# ---------------- Case 6: Batch size > 1 ----------------
print("Case 6: Batch size > 1")
N, C, H, W = 2, 2, 4, 4
kernel_h, kernel_w = 2, 2
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1
pad_h, pad_w = 0, 0

height_col = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
width_col = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

data = np.random.random((N, C * block_size, L)).astype(np.float32)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape,
                     dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr, dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

# ---------------- Case 7: Combined - stride, padding, dilation ----------------
print("Case 7: Combined - stride, padding, dilation")
N, C, H, W = 2, 2, 8, 8
kernel_h, kernel_w = 3, 3
stride_h, stride_w = 2, 2
dilation_h, dilation_w = 2, 2
pad_h, pad_w = 2, 2

height_col = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
width_col = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

data = np.random.random((N, C * block_size, L)).astype(np.float32)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape,
                     dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr, dilations=np.array([dilation_h, dilation_w]),
                     pads=np.array([pad_h, pad_w, pad_h, pad_w]),
                     strides=np.array([stride_h, stride_w]))
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

# ---------------- Case 8: Default parameters (no dilations, pads, strides specified) ----------------
print("Case 8: Default parameters")
N, C, H, W = 1, 1, 5, 5
kernel_h, kernel_w = 2, 2

# Default: stride=1, dilation=1, pad=0
height_col = (H - kernel_h) // 1 + 1
width_col = (W - kernel_w) // 1 + 1
L = height_col * width_col
block_size = kernel_h * kernel_w

data = np.random.random((N, C * block_size, L)).astype(np.float32)
image_shape = np.array([H, W], dtype=np.int64)
block_shape = np.array([kernel_h, kernel_w], dtype=np.int64)

start_time = time.time()
np_out = onnx_col2im(data, image_shape, block_shape)
print("NumPy:", time.time() - start_time, "seconds")

col2im_op = Col2imOp(mgr)  # Use defaults
start_time = time.time()
kp_out = col2im_op.run(data, image_shape, block_shape)[0]
print(f"{col2im_op}:", time.time() - start_time, "seconds")

print(f"Input shape: {data.shape}, Output shape: {kp_out.shape}")
print("Y allclose:", np.allclose(np_out, kp_out, rtol=1e-4, atol=1e-4))
print("Max error:", np.abs(np_out - kp_out).max())
print("----")

print("All tests completed!")