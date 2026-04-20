import sys
from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_conv_optimized import ConvOp


def _make_ind(dim, shape):
    m = np.empty(shape, dtype=np.int64)
    ind = [slice(0, shape[i]) for i in range(len(shape))]
    new_shape = [1] * len(shape)
    new_shape[dim] = shape[dim]
    first = np.arange(shape[dim]).reshape(new_shape)
    m[tuple(ind)] = first
    return m


def im2col_fast(X, kernel_shape, pads, strides):
    n_dims = len(kernel_shape)
    m, n_C = X.shape[:2]

    kernel_size = np.prod(kernel_shape)
    shape_out = []
    for i, dim in enumerate(kernel_shape):
        dx = X.shape[2 + i]
        shape_out.append((dx + pads[i] + pads[i + n_dims] - dim) // strides[i] + 1)

    indices = []
    for i in range(len(shape_out)):
        kind = _make_ind(i, kernel_shape)
        iind = _make_ind(i, shape_out) * strides[i]
        index = np.tile(kind.ravel(), n_C).reshape(-1, 1) + iind.reshape(1, -1)
        indices.append(index)

    d = np.repeat(np.arange(n_C), kernel_size).reshape(-1, 1)

    nc = [(0, 0)] * 2
    padding = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
    X_padded = np.pad(X, tuple(nc) + tuple(padding), mode="constant")

    getitem = (slice(0, m), d, *indices)
    cols = X_padded[getitem]  # type: ignore[index]
    conc_cols = np.concatenate(cols, axis=-1)
    return conc_cols, tuple(shape_out)


def _conv_implementation_im2col(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides):
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]
    kernel_shape = tuple(kernel_shape)

    if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
        raise ValueError(
            f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}, "
            f"W should be {(W.shape[0], X.shape[1] // group, np.prod(W.shape[1:]) // X.shape[1] * group)}."
        )
    if group > 1:
        res = []
        td = 0
        mg = W.shape[0] // group
        dw = W.shape[1]

        for b in range(X.shape[0]):
            for g in range(group):
                gx = X[b : b + 1, g * dw : (g + 1) * dw]
                gw = W[g * mg : (g + 1) * mg]
                try:
                    cv = _conv_implementation_im2col(
                        gx,
                        gw,
                        None,
                        auto_pad,
                        dilations,
                        1,
                        kernel_shape,
                        pads,
                        strides,
                    )
                except (ValueError, RuntimeError) as e:
                    raise ValueError(
                        f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={g}/{group}, "
                        f"gx.shape={gx.shape}, gw.shape={gw.shape}, auto_pad={auto_pad}, "
                        f"dilations={dilations}, kernel_shape={kernel_shape}, pads={pads}, "
                        f"strides={strides}."
                    ) from e
                if b == 0:
                    td += cv.shape[1]
                res.append((b, cv))

        new_shape = [X.shape[0], *list(res[0][1].shape[1:])]
        new_shape[1] = td
        final = np.zeros(tuple(new_shape), dtype=res[0][1].dtype)
        p = 0
        for b, cv in res:
            final[b : b + 1, p : p + cv.shape[1]] = cv
            p += cv.shape[1]
            if p >= final.shape[1]:
                p = 0
        if B is not None:
            new_shape = [1 for s in final.shape]
            new_shape[1] = B.shape[0]
            b = B.reshape(tuple(new_shape))
            final += b
        return final

    if dilations[0] != 1 or min(dilations) != max(dilations):
        # Let's compute the dilated kernel.
        nd = len(dilations)
        new_kernel_shape = []
        new_shape = list(W.shape[:-nd])
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            new_shape.append(W.shape[di] + (W.shape[di] - 1) * (d - 1))
            new_kernel_shape.append(kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1))
        new_w = np.zeros(tuple(new_shape), dtype=W.dtype)
        indices = [slice(0, new_w.shape[0]), slice(0, new_w.shape[1])]
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            indices.append(slice(0, new_w.shape[di], d))
        new_w[tuple(indices)] = W
        W = new_w
        kernel_shape = new_kernel_shape

    if auto_pad == "VALID":
        pads = [0] * ((len(X.shape) - 2) * 2)
    elif auto_pad in {"SAME_LOWER", "SAME_UPPER"}:
        head = []
        tail = []
        for i in range(len(X.shape) - 2):
            d = X.shape[2 + i]  # 空间维度，而非 batch/channel
            target_size = (d + strides[i] - 1) // strides[i]
            pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
            if auto_pad == "SAME_LOWER":
                pad_head = (pad_needed + 1) // 2
            else:
                pad_head = pad_needed // 2
            pad_tail = pad_needed - pad_head
            head.append(pad_head)
            tail.append(pad_tail)
        pads = head + tail

    c2, out_shape = im2col_fast(X, kernel_shape, pads, strides)
    w_reshaped = W.reshape((-1, c2.shape[0]))
    mul = w_reshaped @ c2
    mul = mul.reshape((W.shape[0], X.shape[0], *out_shape))
    perm = (1, 0, *tuple(np.arange(len(X.shape) - 2) + 2))
    mul = mul.transpose(perm)

    if B is not None:
        if B.size == 1:
            return mul + B
        new_shape = [1] * len(mul.shape)
        new_shape[1] = -1
        mul += B.reshape(tuple(new_shape))
    return mul


# 供测试用的 baseline 入口

def onnx_reference_conv(X, W, B=None, auto_pad="NOTSET", dilations=None, group=1,
                        kernel_shape=None, pads=None, strides=None):
    return _conv_implementation_im2col(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides).astype(X.dtype)


# ---- 测试执行 ----
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

conv_op = ConvOp(mgr)

# Case 1: 1D Conv, basic, no padding
print("Case 1: 1D Conv, basic, no padding")
numpy_in = np.random.uniform(-1, 1, (2, 3, 32)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.kernel_shape = None
conv_op.strides = None
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
conv_op.group = 1
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: 1D Conv, with stride
print("Case 2: 1D Conv, with stride")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, strides=[2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2]
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: 1D Conv, with padding
print("Case 3: 1D Conv, with padding")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, pads=[1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = None
conv_op.pads = [1, 1]
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 1D Conv, with dilation
print("Case 4: 1D Conv, with dilation")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, dilations=[2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = None
conv_op.pads = None
conv_op.dilations = [2]
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: 1D Conv, with bias
print("Case 5: 1D Conv, with bias")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
numpy_b = np.random.uniform(-1, 1, (4,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, numpy_b)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = None
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w, numpy_b)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: 1D Conv, auto_pad SAME_UPPER
print("Case 6: 1D Conv, auto_pad SAME_UPPER")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, auto_pad="SAME_UPPER", strides=[2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2]
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "SAME_UPPER"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 7: 1D Conv, auto_pad SAME_LOWER
print("Case 7: 1D Conv, auto_pad SAME_LOWER")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, auto_pad="SAME_LOWER", strides=[2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2]
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "SAME_LOWER"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 8: 1D Conv, auto_pad VALID
# VALID 等价于无 padding（pads 全 0），输出长度 = (in - k) / s + 1
print("Case 8: 1D Conv, auto_pad VALID")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, auto_pad="VALID")
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = None
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "VALID"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 9: 2D Conv, basic
print("Case 9: 2D Conv, basic")
numpy_in = np.random.uniform(-1, 1, (2, 3, 32, 32)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = None
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 10: 2D Conv, with stride and padding
print("Case 10: 2D Conv, with stride and padding")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, strides=[2, 2], pads=[1, 1, 1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2, 2]
conv_op.pads = [1, 1, 1, 1]
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 11: 2D Conv, with bias
print("Case 11: 2D Conv, with bias")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
numpy_b = np.random.uniform(-1, 1, (8,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, numpy_b, strides=[2, 2], pads=[1, 1, 1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2, 2]
conv_op.pads = [1, 1, 1, 1]
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w, numpy_b)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 12: 2D Conv, auto_pad SAME_UPPER
print("Case 12: 2D Conv, auto_pad SAME_UPPER")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, auto_pad="SAME_UPPER", strides=[2, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2, 2]
conv_op.pads = None
conv_op.dilations = None
conv_op.auto_pad = "SAME_UPPER"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 13: 2D Conv, with dilation
print("Case 13: 2D Conv, with dilation")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, dilations=[2, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = None
conv_op.pads = None
conv_op.dilations = [2, 2]
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 14: 2D Conv, asymmetric padding
print("Case 14: 2D Conv, asymmetric padding")
numpy_in = np.random.uniform(-1, 1, (2, 3, 64, 64)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, strides=[2, 2], pads=[0, 1, 2, 3])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2, 2]
conv_op.pads = [0, 1, 2, 3]
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 15: 3D Conv, basic  (ssbo 版不支持 3D，跳过)
print("Case 15: 3D Conv, basic")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16, 16, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

try:
    start_time = time.time()
    conv_op.strides = None
    conv_op.pads = None
    conv_op.dilations = None
    conv_op.auto_pad = "NOTSET"
    kp_out = conv_op.run(numpy_in, numpy_w)[0]
    print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
    print("Max error:", np.abs(numpy_out - kp_out).max())
    print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
except NotImplementedError as e:
    print(f"SKIP (ssbo does not support 3D): {e}")
print()

# Case 16: 3D Conv, with stride and padding  (ssbo 版不支持 3D，跳过)
print("Case 16: 3D Conv, with stride and padding")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16, 16, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, strides=[2, 2, 2], pads=[1, 1, 1, 1, 1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

try:
    start_time = time.time()
    conv_op.strides = [2, 2, 2]
    conv_op.pads = [1, 1, 1, 1, 1, 1]
    conv_op.dilations = None
    conv_op.auto_pad = "NOTSET"
    kp_out = conv_op.run(numpy_in, numpy_w)[0]
    print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
    print("Max error:", np.abs(numpy_out - kp_out).max())
    print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
except NotImplementedError as e:
    print(f"SKIP (ssbo does not support 3D): {e}")
print()

# Case 17: 1D group=2 basic
print("Case 17: 1D Conv, group=2 basic")
numpy_in = np.random.uniform(-1, 1, (2, 6, 64)).astype(np.float32)
# W shape [Cout, Cin/group, K]
numpy_w = np.random.uniform(-1, 1, (8, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, group=2, pads=[1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

conv_op_g = ConvOp(mgr)
conv_op_g.group = 2
conv_op_g.auto_pad = "NOTSET"
conv_op_g.kernel_shape = None
conv_op_g.strides = None
conv_op_g.pads = [1, 1]
conv_op_g.dilations = None
start_time = time.time()
kp_out = conv_op_g.run(numpy_in, numpy_w)[0]
print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 18: 2D group=2 stride+pad
print("Case 18: 2D Conv, group=2 stride+pad")
numpy_in = np.random.uniform(-1, 1, (2, 6, 32, 33)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, group=2, strides=[2, 1], pads=[1, 1, 2, 0])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

conv_op_g = ConvOp(mgr)
conv_op_g.group = 2
conv_op_g.auto_pad = "NOTSET"
conv_op_g.kernel_shape = None
conv_op_g.strides = [2, 1]
conv_op_g.pads = [1, 1, 2, 0]
conv_op_g.dilations = None
start_time = time.time()
kp_out = conv_op_g.run(numpy_in, numpy_w)[0]
print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 19: 2D group=4 with dilation
print("Case 19: 2D Conv, group=4 dilation")
numpy_in = np.random.uniform(-1, 1, (1, 8, 28, 29)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (12, 2, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, group=4, dilations=[2, 1], strides=[1, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

conv_op_g = ConvOp(mgr)
conv_op_g.group = 4
conv_op_g.auto_pad = "NOTSET"
conv_op_g.kernel_shape = None
conv_op_g.strides = [1, 2]
conv_op_g.pads = None
conv_op_g.dilations = [2, 1]
start_time = time.time()
kp_out = conv_op_g.run(numpy_in, numpy_w)[0]
print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 20: 3D group=2 bias+pad+stride  (ssbo 版不支持 3D，跳过)
print("Case 20: 3D Conv, group=2 bias+pad+stride")
numpy_in = np.random.uniform(-1, 1, (1, 4, 16, 15, 14)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (6, 2, 3, 3, 2)).astype(np.float32)
numpy_b = np.random.uniform(-1, 1, (6,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, numpy_b, group=2, strides=[2, 1, 1], pads=[1, 0, 2, 0, 1, 0])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

conv_op_g = ConvOp(mgr)
conv_op_g.group = 2
conv_op_g.auto_pad = "NOTSET"
conv_op_g.kernel_shape = None
conv_op_g.strides = [2, 1, 1]
conv_op_g.pads = [1, 0, 2, 0, 1, 0]
conv_op_g.dilations = None
try:
    start_time = time.time()
    kp_out = conv_op_g.run(numpy_in, numpy_w, numpy_b)[0]
    print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
    print("Max error:", np.abs(numpy_out - kp_out).max())
    print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
except NotImplementedError as e:
    print(f"SKIP (ssbo does not support 3D): {e}")
print()

