import sys
from kp import Manager
import numpy as np
import time
from kp_onnx.kop_conv_integer import ConvIntegerOp


def onnx_reference_conv_integer(X, W, x_zero_point=None, w_zero_point=None, auto_pad="NOTSET", dilations=None, group=1,
                                kernel_shape=None, pads=None, strides=None):
    """纯 NumPy 的 ConvInteger 参考实现，支持 group 和零点广播。"""
    X = np.asarray(X).astype(np.int32)
    W = np.asarray(W).astype(np.int32)

    # ONNX: x_zero_point 可为标量或 [Cin]；w_zero_point 可为标量或 [Cout]
    if x_zero_point is not None:
        xzp = np.asarray(x_zero_point, dtype=np.int32)
        if xzp.ndim == 1 and xzp.shape[0] == X.shape[1]:
            xzp = xzp.reshape((1, X.shape[1]) + (1,) * (X.ndim - 2))
        X = X - xzp
    if w_zero_point is not None:
        wzp = np.asarray(w_zero_point, dtype=np.int32)
        if wzp.ndim == 1 and wzp.shape[0] == W.shape[0]:
            wzp = wzp.reshape((W.shape[0],) + (1,) * (W.ndim - 1))
        W = W - wzp

    if dilations is None:
        dilations = [1 for _ in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for _ in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for _ in X.shape[2:]]

    if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
        raise ValueError("Shape inconsistencies")

    # group > 1: 逐组计算再拼接
    if group > 1:
        cinpg = X.shape[1] // group
        coutpg = W.shape[0] // group
        ys = []
        for g in range(group):
            xs = X[:, g*cinpg:(g+1)*cinpg]
            ws = W[g*coutpg:(g+1)*coutpg]
            ys.append(onnx_reference_conv_integer(xs, ws, None, None, auto_pad, dilations, 1, kernel_shape, pads, strides))
        return np.concatenate(ys, axis=1).astype(np.int32)

    # dilation: 扩张卷积核
    if len(dilations) and (min(dilations) != 1 or max(dilations) != 1):
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

    # auto_pad
    if auto_pad in {"SAME_LOWER", "SAME_UPPER", "VALID"}:
        head = []
        tail = []
        for i in range(len(X.shape) - 2):
            d = X.shape[i + 2]
            target_size = (d + strides[i] - 1) // strides[i]
            pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
            if auto_pad == "SAME_LOWER":
                pad_head = (pad_needed + 1) // 2
            elif auto_pad == "SAME_UPPER":
                pad_head = pad_needed // 2
            else:  # VALID
                pad_head = 0
                pad_needed = 0
            pad_tail = pad_needed - pad_head
            head.append(pad_head)
            tail.append(pad_tail)
        pads = head + tail

    # 1D
    if len(X.shape) == 3:
        sN, sC, sH = X.shape
        (kh,) = kernel_shape
        (sth,) = strides
        h_out = int(((sH - kh + pads[0] + pads[1]) / sth) + 1)
        res = np.zeros((X.shape[0], W.shape[0], h_out), dtype=np.int64)
        for n in range(sN):
            for nw in range(W.shape[0]):
                for c in range(sC):
                    w = W[nw, c]
                    for hr in range(h_out):
                        i = hr * sth - pads[0]
                        for k in range(kh):
                            ih = i + k
                            if 0 <= ih < sH:
                                res[n, nw, hr] += int(X[n, c, ih]) * int(w[k])
        return res.astype(np.int32)

    # 2D
    if len(X.shape) == 4:
        sN, sC, sH, sW = X.shape
        kh, kw = kernel_shape
        sth, stw = strides
        h_out = int(((sH - kh + pads[0] + pads[2]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[3]) / stw) + 1)
        res = np.zeros((X.shape[0], W.shape[0], h_out, w_out), dtype=np.int64)
        for n in range(sN):
            for nw in range(W.shape[0]):
                for c in range(sC):
                    w = W[nw, c]
                    for hr in range(h_out):
                        for wr in range(w_out):
                            i = hr * sth - pads[0]
                            j = wr * stw - pads[1]
                            for kh_idx in range(kh):
                                for kw_idx in range(kw):
                                    ih = i + kh_idx
                                    iw = j + kw_idx
                                    if 0 <= ih < sH and 0 <= iw < sW:
                                        res[n, nw, hr, wr] += int(X[n, c, ih, iw]) * int(w[kh_idx, kw_idx])
        return res.astype(np.int32)

    # 3D
    if len(X.shape) == 5:
        sN, sC, sH, sW, sD = X.shape
        kh, kw, kd = kernel_shape
        sth, stw, std = strides
        h_out = int(((sH - kh + pads[0] + pads[3]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[4]) / stw) + 1)
        d_out = int(((sD - kd + pads[2] + pads[5]) / std) + 1)
        res = np.zeros((X.shape[0], W.shape[0], h_out, w_out, d_out), dtype=np.int64)
        for n in range(sN):
            for nw in range(W.shape[0]):
                for c in range(sC):
                    w = W[nw, c]
                    for hr in range(h_out):
                        for wr in range(w_out):
                            for dr in range(d_out):
                                i = hr * sth - pads[0]
                                j = wr * stw - pads[1]
                                k = dr * std - pads[2]
                                for kh_idx in range(kh):
                                    for kw_idx in range(kw):
                                        for kd_idx in range(kd):
                                            ih = i + kh_idx
                                            iw = j + kw_idx
                                            id = k + kd_idx
                                            if 0 <= ih < sH and 0 <= iw < sW and 0 <= id < sD:
                                                res[n, nw, hr, wr, dr] += int(X[n, c, ih, iw, id]) * int(w[kh_idx, kw_idx, kd_idx])
        return res.astype(np.int32)

    raise RuntimeError(f"Unsupported shape: {X.shape}")


# 测试执行部分（与 kop_conv_test.py 风格对齐）
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

conv_op = ConvIntegerOp(mgr)

# 随机源（小范围避免 int32 溢出）
rng = np.random.default_rng(123)

# Case 1: 1D ConvInteger, basic, no padding
print("Case 1: 1D ConvInteger, basic, no padding")
X = rng.integers(-5, 6, size=(2, 3, 32), dtype=np.int32)
W = rng.integers(-5, 6, size=(4, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.kernel_shape = None; conv_op.strides = None; conv_op.pads = None; conv_op.dilations = None; conv_op.auto_pad = "NOTSET"; conv_op.group = 1
start_time = time.time(); kp_out = conv_op.run(X, W)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: 1D stride
print("Case 2: 1D stride")
X = rng.integers(-5, 6, size=(2, 3, 64), dtype=np.int32)
W = rng.integers(-5, 6, size=(4, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, strides=[2]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = [2]; conv_op.pads = None; conv_op.dilations = None; conv_op.auto_pad = "NOTSET"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: 1D padding
print("Case 3: 1D padding")
X = rng.integers(-5, 6, size=(2, 3, 64), dtype=np.int32)
W = rng.integers(-5, 6, size=(4, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, pads=[1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = None; conv_op.pads = [1, 1]; conv_op.dilations = None; conv_op.auto_pad = "NOTSET"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 1D dilation
print("Case 4: 1D dilation")
X = rng.integers(-4, 5, size=(2, 3, 64), dtype=np.int32)
W = rng.integers(-4, 5, size=(4, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, dilations=[2]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = None; conv_op.pads = None; conv_op.dilations = [2]; conv_op.auto_pad = "NOTSET"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: 1D auto_pad SAME_UPPER
print("Case 5: 1D auto_pad SAME_UPPER")
X = rng.integers(-4, 5, size=(2, 3, 64), dtype=np.int32)
W = rng.integers(-4, 5, size=(4, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, auto_pad="SAME_UPPER", strides=[2]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = [2]; conv_op.pads = None; conv_op.dilations = None; conv_op.auto_pad = "SAME_UPPER"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: 1D auto_pad VALID
print("Case 6: 1D auto_pad VALID")
X = rng.integers(-4, 5, size=(2, 3, 64), dtype=np.int32)
W = rng.integers(-4, 5, size=(4, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, auto_pad="VALID"); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = None; conv_op.pads = None; conv_op.dilations = None; conv_op.auto_pad = "VALID"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 7: 1D group=2
print("Case 7: 1D group=2")
X = rng.integers(-3, 4, size=(2, 6, 64), dtype=np.int32)
W = rng.integers(-3, 4, size=(8, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, group=2, pads=[1, 1]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op_g = ConvIntegerOp(mgr); conv_op_g.group = 2; conv_op_g.auto_pad = "NOTSET"; conv_op_g.kernel_shape = None; conv_op_g.strides = None; conv_op_g.pads = [1, 1]; conv_op_g.dilations = None
start_time = time.time(); kp_out = conv_op_g.run(X, W)[0]; print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 8: 2D basic
print("Case 8: 2D basic")
X = rng.integers(-3, 4, size=(2, 3, 32, 33), dtype=np.int32)
W = rng.integers(-3, 4, size=(8, 3, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.auto_pad = "NOTSET"; conv_op.strides = None; conv_op.pads = None; conv_op.dilations = None
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 9: 2D stride+pad
print("Case 9: 2D stride+pad")
X = rng.integers(-3, 4, size=(2, 3, 64, 64), dtype=np.int32)
W = rng.integers(-3, 4, size=(8, 3, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, strides=[2, 2], pads=[1, 1, 1, 1]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = [2, 2]; conv_op.pads = [1, 1, 1, 1]; conv_op.dilations = None; conv_op.auto_pad = "NOTSET"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 10: 2D dilation
print("Case 10: 2D dilation")
X = rng.integers(-3, 4, size=(1, 4, 32, 31), dtype=np.int32)
W = rng.integers(-3, 4, size=(6, 4, 3, 2), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, dilations=[2, 1], strides=[1, 2]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = [1, 2]; conv_op.pads = None; conv_op.dilations = [2, 1]; conv_op.auto_pad = "NOTSET"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 11: 2D asymmetric padding
print("Case 11: 2D asymmetric padding")
X = rng.integers(-3, 4, size=(2, 3, 64, 64), dtype=np.int32)
W = rng.integers(-3, 4, size=(8, 3, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, strides=[2, 1], pads=[0, 1, 2, 3]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = [2, 1]; conv_op.pads = [0, 1, 2, 3]; conv_op.dilations = None; conv_op.auto_pad = "NOTSET"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 12: 2D auto_pad SAME_LOWER
print("Case 12: 2D auto_pad SAME_LOWER")
X = rng.integers(-3, 4, size=(2, 3, 64, 64), dtype=np.int32)
W = rng.integers(-3, 4, size=(8, 3, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, auto_pad="SAME_LOWER", strides=[2, 2]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = [2, 2]; conv_op.pads = None; conv_op.dilations = None; conv_op.auto_pad = "SAME_LOWER"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 13: 2D group=4 + dilations
print("Case 13: 2D group=4 + dilations")
X = rng.integers(-2, 3, size=(1, 8, 28, 29), dtype=np.int32)
W = rng.integers(-2, 3, size=(12, 2, 3, 3), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, group=4, dilations=[2, 1], strides=[1, 2]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op_g = ConvIntegerOp(mgr); conv_op_g.group = 4; conv_op_g.auto_pad = "NOTSET"; conv_op_g.kernel_shape = None; conv_op_g.strides = [1, 2]; conv_op_g.pads = None; conv_op_g.dilations = [2, 1]
start_time = time.time(); kp_out = conv_op_g.run(X, W)[0]; print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 14: 3D basic
print("Case 14: 3D basic")
X = rng.integers(-2, 3, size=(1, 3, 10, 9, 8), dtype=np.int32)
W = rng.integers(-2, 3, size=(4, 3, 3, 3, 2), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.auto_pad = "NOTSET"; conv_op.strides = None; conv_op.pads = None; conv_op.dilations = None
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 15: 3D stride+pad
print("Case 15: 3D stride+pad")
X = rng.integers(-2, 3, size=(1, 3, 16, 15, 14), dtype=np.int32)
W = rng.integers(-2, 3, size=(6, 3, 3, 3, 2), dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, strides=[2, 1, 1], pads=[1, 0, 2, 0, 1, 0]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op.strides = [2, 1, 1]; conv_op.pads = [1, 0, 2, 0, 1, 0]; conv_op.dilations = None; conv_op.auto_pad = "NOTSET"
start_time = time.time(); kp_out = conv_op.run(X, W)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 16: zero points scalar
print("Case 16: zero points scalar")
X = rng.integers(0, 16, size=(2, 3, 16), dtype=np.int32)
W = rng.integers(0, 16, size=(4, 3, 3), dtype=np.int32)
xzp = np.array(8, dtype=np.int32); wzp = np.array(7, dtype=np.int32)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, x_zero_point=xzp, w_zero_point=wzp, pads=[1, 1]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op = ConvIntegerOp(mgr); conv_op.pads = [1, 1]
start_time = time.time(); kp_out = conv_op.run(X, W, xzp, wzp)[0]; print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 17: per-channel zero points
print("Case 17: per-channel zero points")
X = rng.integers(0, 16, size=(1, 4, 20, 21), dtype=np.int32)
W = rng.integers(0, 16, size=(6, 2, 3, 3), dtype=np.int32)
# group=2，Cin/group=2 -> xzp shape (4,)；wzp shape (6,)
xzp = rng.integers(0, 16, size=(4,), dtype=np.int32)
wzp = rng.integers(0, 16, size=(6,), dtype=np.int32)
xzp_exp = xzp.reshape(1, 4, 1, 1)
wzp_exp = wzp.reshape(6, 1, 1, 1)
start_time = time.time(); numpy_out = onnx_reference_conv_integer(X, W, x_zero_point=xzp_exp, w_zero_point=wzp_exp, group=2, strides=[2, 1], pads=[1, 0, 2, 0]); print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")
conv_op_g = ConvIntegerOp(mgr); conv_op_g.group = 2; conv_op_g.strides = [2, 1]; conv_op_g.pads = [1, 0, 2, 0]
start_time = time.time(); kp_out = conv_op_g.run(X, W, xzp_exp, wzp_exp)[0]; print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()
