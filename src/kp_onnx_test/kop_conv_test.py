import sys
from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_conv import ConvOp


def onnx_reference_conv(X, W, B=None, auto_pad="NOTSET", dilations=None, group=1,
                        kernel_shape=None, pads=None, strides=None):
    """参考实现：基于ONNX规范的Conv，支持 group"""
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]

    if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
        raise ValueError(f"Shape inconsistencies")

    # group > 1: 拆分输入/权重/可选bias，逐组计算并在通道维度拼接
    if group > 1:
        cinpg = X.shape[1] // group
        coutpg = W.shape[0] // group
        ys = []
        for g in range(group):
            xs = X[:, g*cinpg:(g+1)*cinpg]
            ws = W[g*coutpg:(g+1)*coutpg]
            bs = None if B is None else B[g*coutpg:(g+1)*coutpg]
            ys.append(onnx_reference_conv(xs, ws, bs, auto_pad, dilations, 1, kernel_shape, pads, strides))
        return np.concatenate(ys, axis=1)

    # Handle dilations by expanding kernel
    if dilations[0] != 1 or min(dilations) != max(dilations):
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

    # Handle auto_pad
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

    # 1D convolution
    if len(X.shape) == 3:
        sN, sC, sH = X.shape
        kh, = kernel_shape
        sth, = strides

        h_out = int(((sH - kh + pads[0] + pads[1]) / sth) + 1)

        res = np.zeros((X.shape[0], W.shape[0], h_out), dtype=X.dtype)
        if B is not None:
            res[:, :, :] += B.reshape((1, -1, 1))

        for n in range(sN):
            for nw in range(W.shape[0]):
                for c in range(sC):
                    w = W[nw, c]
                    for hr in range(h_out):
                        i = hr * sth - pads[0]
                        for k in range(kh):
                            ih = i + k
                            if 0 <= ih < sH:
                                res[n, nw, hr] += X[n, c, ih] * w[k]

        return res

    # 2D convolution
    if len(X.shape) == 4:
        sN, sC, sH, sW = X.shape
        kh, kw = kernel_shape
        sth, stw = strides

        h_out = int(((sH - kh + pads[0] + pads[2]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[3]) / stw) + 1)

        res = np.zeros((X.shape[0], W.shape[0], h_out, w_out), dtype=X.dtype)
        if B is not None:
            res[:, :, :, :] += B.reshape((1, -1, 1, 1))

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
                                        res[n, nw, hr, wr] += X[n, c, ih, iw] * w[kh_idx, kw_idx]

        return res

    # 3D convolution
    if len(X.shape) == 5:
        sN, sC, sH, sW, sD = X.shape
        kh, kw, kd = kernel_shape
        sth, stw, std = strides

        h_out = int(((sH - kh + pads[0] + pads[3]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[4]) / stw) + 1)
        d_out = int(((sD - kd + pads[2] + pads[5]) / std) + 1)

        res = np.zeros((X.shape[0], W.shape[0], h_out, w_out, d_out), dtype=X.dtype)
        if B is not None:
            res[:, :, :, :, :] += B.reshape((1, -1, 1, 1, 1))

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
                                                res[n, nw, hr, wr, dr] += X[n, c, ih, iw, id] * w[kh_idx, kw_idx, kd_idx]

        return res

    raise RuntimeError(f"Unsupported shape: {X.shape}")


# 测试执行部分
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

# Case 15: 3D Conv, basic
print("Case 15: 3D Conv, basic")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16, 16, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3, 3, 3)).astype(np.float32)
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

# Case 16: 3D Conv, with stride and padding
print("Case 16: 3D Conv, with stride and padding")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16, 16, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv(numpy_in, numpy_w, strides=[2, 2, 2], pads=[1, 1, 1, 1, 1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
conv_op.strides = [2, 2, 2]
conv_op.pads = [1, 1, 1, 1, 1, 1]
conv_op.dilations = None
conv_op.auto_pad = "NOTSET"
kp_out = conv_op.run(numpy_in, numpy_w)[0]
print(f"{conv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
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

# Case 20: 3D group=2 bias+pad+stride
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
start_time = time.time()
kp_out = conv_op_g.run(numpy_in, numpy_w, numpy_b)[0]
print(f"{conv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

