import sys
from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_conv_transpose import ConvTransposeOp


def onnx_reference_conv_transpose(X, W, B=None, auto_pad="NOTSET", dilations=None, group=1,
                                   kernel_shape=None, output_padding=None, output_shape=None, pads=None, strides=None):
    """参考实现：基于ONNX规范的ConvTranspose，支持 group"""
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if output_padding is None:
        output_padding = [0 for s in X.shape[2:]]
    if strides is None:
        strides = [1 for s in X.shape[2:]]

    dims = len(X.shape[2:])

    # 推导 output_shape 和 pads
    if pads is None and auto_pad not in {"SAME_UPPER", "SAME_LOWER"}:
        pads = [0 for _ in range(2 * dims)]

    if pads is None:
        # auto_pad SAME_*
        if output_shape is None:
            output_shape = [X.shape[i + 2] * strides[i] for i in range(dims)]
        total_padding = [
            strides[i] * (X.shape[i + 2] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
            for i in range(dims)
        ]
        pads_1 = []
        pads_2 = []
        for i in range(dims):
            if auto_pad == "SAME_UPPER":
                pads_1.append(total_padding[i] // 2)
                pads_2.append(total_padding[i] - (total_padding[i] // 2))
            else:
                pads_1.append(total_padding[i] - (total_padding[i] // 2))
                pads_2.append(total_padding[i] // 2)
        pads = pads_1 + pads_2
    else:
        if output_shape is None:
            output_shape = [
                strides[i] * (X.shape[i + 2] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - (pads[i] + pads[i + dims])
                for i in range(dims)
            ]

    if X.shape[1] != W.shape[0] or W.shape[0] % group != 0:
        raise ValueError(f"Shape inconsistencies")

    # group > 1: 拆分输入/权重/可选bias，逐组计算并在通道维度拼接
    if group > 1:
        cinpg = X.shape[1] // group
        coutpg = W.shape[1]
        ys = []
        for g in range(group):
            xs = X[:, g*cinpg:(g+1)*cinpg]
            ws = W[g*cinpg:(g+1)*cinpg]
            bs = None if B is None else B[g*coutpg:(g+1)*coutpg]
            ys.append(onnx_reference_conv_transpose(xs, ws, bs, auto_pad, dilations, 1, kernel_shape, output_padding, output_shape, pads, strides))
        return np.concatenate(ys, axis=1)

    # 1D ConvTranspose
    if len(X.shape) == 3:
        N, Cin, iH = X.shape
        _, Cout_per_group, kH = W.shape
        oH = output_shape[0]
        pad_head, pad_tail = pads[0], pads[1]

        res = np.zeros((N, Cout_per_group, oH), dtype=X.dtype)
        if B is not None:
            res += B.reshape((1, -1, 1))

        for n in range(N):
            for oc in range(Cout_per_group):
                for ci in range(Cin):
                    w = W[ci, oc]
                    for i_idx in range(iH):
                        for k_idx in range(kH):
                            o_idx = i_idx * strides[0] + k_idx * dilations[0] - pad_head
                            if 0 <= o_idx < oH:
                                res[n, oc, o_idx] += X[n, ci, i_idx] * w[k_idx]
        return res

    # 2D ConvTranspose
    if len(X.shape) == 4:
        N, Cin, iH, iW = X.shape
        _, Cout_per_group, kH, kW = W.shape
        oH, oW = output_shape
        pad_h_head, pad_w_head = pads[0], pads[1]

        res = np.zeros((N, Cout_per_group, oH, oW), dtype=X.dtype)
        if B is not None:
            res += B.reshape((1, -1, 1, 1))

        for n in range(N):
            for oc in range(Cout_per_group):
                for ci in range(Cin):
                    w = W[ci, oc]
                    for ih in range(iH):
                        for iw in range(iW):
                            for kh in range(kH):
                                for kw in range(kW):
                                    oh = ih * strides[0] + kh * dilations[0] - pad_h_head
                                    ow = iw * strides[1] + kw * dilations[1] - pad_w_head
                                    if 0 <= oh < oH and 0 <= ow < oW:
                                        res[n, oc, oh, ow] += X[n, ci, ih, iw] * w[kh, kw]
        return res

    # 3D ConvTranspose
    if len(X.shape) == 5:
        N, Cin, iD, iH, iW = X.shape
        _, Cout_per_group, kD, kH, kW = W.shape
        oD, oH, oW = output_shape
        pad_d_head, pad_h_head, pad_w_head = pads[0], pads[1], pads[2]

        res = np.zeros((N, Cout_per_group, oD, oH, oW), dtype=X.dtype)
        if B is not None:
            res += B.reshape((1, -1, 1, 1, 1))

        for n in range(N):
            for oc in range(Cout_per_group):
                for ci in range(Cin):
                    w = W[ci, oc]
                    for id_idx in range(iD):
                        for ih in range(iH):
                            for iw in range(iW):
                                for kd in range(kD):
                                    for kh in range(kH):
                                        for kw in range(kW):
                                            od = id_idx * strides[0] + kd * dilations[0] - pad_d_head
                                            oh = ih * strides[1] + kh * dilations[1] - pad_h_head
                                            ow = iw * strides[2] + kw * dilations[2] - pad_w_head
                                            if 0 <= od < oD and 0 <= oh < oH and 0 <= ow < oW:
                                                res[n, oc, od, oh, ow] += X[n, ci, id_idx, ih, iw] * w[kd, kh, kw]
        return res

    raise RuntimeError(f"Unsupported shape: {X.shape}")


# 测试执行部分
device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

deconv_op = ConvTransposeOp(mgr)

# Case 1: 1D ConvTranspose, basic, no padding
print("Case 1: 1D ConvTranspose, basic, no padding")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 4, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.kernel_shape = None
deconv_op.strides = None
deconv_op.pads = None
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.group = 1
deconv_op.output_padding = None
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 2: 1D ConvTranspose, with stride
print("Case 2: 1D ConvTranspose, with stride")
numpy_in = np.random.uniform(-1, 1, (2, 3, 17)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 4, 5)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, strides=[2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2]
deconv_op.pads = None
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 3: 1D ConvTranspose, with padding
print("Case 3: 1D ConvTranspose, with padding")
numpy_in = np.random.uniform(-1, 1, (1, 2, 9)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, pads=[1, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = None
deconv_op.pads = [1, 2]
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 4: 1D ConvTranspose, with dilation
print("Case 4: 1D ConvTranspose, with dilation")
numpy_in = np.random.uniform(-1, 1, (1, 2, 10)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (2, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, dilations=[2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = None
deconv_op.pads = None
deconv_op.dilations = [2]
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 5: 1D ConvTranspose, with bias
print("Case 5: 1D ConvTranspose, with bias")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 4, 3)).astype(np.float32)
numpy_b = np.random.uniform(-1, 1, (4,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, numpy_b)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = None
deconv_op.pads = None
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w, numpy_b)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 6: 1D ConvTranspose, with output_shape
print("Case 6: 1D ConvTranspose, with output_shape")
numpy_in = np.random.uniform(-1, 1, (1, 2, 7)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, strides=[2], pads=[1, 1], output_shape=[15])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2]
deconv_op.pads = [1, 1]
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_shape = [15]
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 7: 1D ConvTranspose, with output_padding
print("Case 7: 1D ConvTranspose, with output_padding")
numpy_in = np.random.uniform(-1, 1, (1, 2, 7)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, strides=[2], pads=[1, 1], output_padding=[1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2]
deconv_op.pads = [1, 1]
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = [1]
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 8: 1D ConvTranspose, auto_pad SAME_UPPER
print("Case 8: 1D ConvTranspose, auto_pad SAME_UPPER")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 4, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, auto_pad="SAME_UPPER", strides=[2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2]
deconv_op.pads = None
deconv_op.dilations = None
deconv_op.auto_pad = "SAME_UPPER"
deconv_op.output_padding = None
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 9: 2D ConvTranspose, basic
print("Case 9: 2D ConvTranspose, basic")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16, 17)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = None
deconv_op.pads = None
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 10: 2D ConvTranspose, with stride and padding
print("Case 10: 2D ConvTranspose, with stride and padding")
numpy_in = np.random.uniform(-1, 1, (1, 3, 16, 17)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 8, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, strides=[2, 1], pads=[1, 2, 1, 0])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2, 1]
deconv_op.pads = [1, 2, 1, 0]
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 11: 2D ConvTranspose, with bias
print("Case 11: 2D ConvTranspose, with bias")
numpy_in = np.random.uniform(-1, 1, (1, 4, 12, 11)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 6, 3, 2)).astype(np.float32)
numpy_b = np.random.uniform(-1, 1, (6,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, numpy_b, dilations=[2, 1], strides=[1, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [1, 2]
deconv_op.pads = None
deconv_op.dilations = [2, 1]
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w, numpy_b)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 12: 2D ConvTranspose, auto_pad SAME_UPPER
print("Case 12: 2D ConvTranspose, auto_pad SAME_UPPER")
numpy_in = np.random.uniform(-1, 1, (2, 3, 16, 17)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 8, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, auto_pad="SAME_UPPER", strides=[2, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2, 2]
deconv_op.pads = None
deconv_op.dilations = None
deconv_op.auto_pad = "SAME_UPPER"
deconv_op.output_padding = None
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 13: 2D ConvTranspose, with dilation
print("Case 13: 2D ConvTranspose, with dilation")
numpy_in = np.random.uniform(-1, 1, (1, 3, 12, 13)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 8, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, dilations=[2, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = None
deconv_op.pads = None
deconv_op.dilations = [2, 2]
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 14: 2D ConvTranspose, asymmetric padding + output_padding
print("Case 14: 2D ConvTranspose, asymmetric padding + output_padding")
numpy_in = np.random.uniform(-1, 1, (1, 3, 9, 10)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (3, 8, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, strides=[2, 2], pads=[0, 1, 2, 3], output_padding=[1, 0])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2, 2]
deconv_op.pads = [0, 1, 2, 3]
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = [1, 0]
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 15: 3D ConvTranspose, basic
print("Case 15: 3D ConvTranspose, basic")
numpy_in = np.random.uniform(-1, 1, (1, 2, 6, 5, 4)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (2, 3, 3, 2, 2)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w)
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = None
deconv_op.pads = None
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
deconv_op.output_shape = None
kp_out = deconv_op.run(numpy_in, numpy_w)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 16: 3D ConvTranspose, with stride and padding
print("Case 16: 3D ConvTranspose, with stride and padding")
numpy_in = np.random.uniform(-1, 1, (1, 4, 5, 6, 4)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 6, 3, 3, 2)).astype(np.float32)
numpy_b = np.random.uniform(-1, 1, (6,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, numpy_b, strides=[2, 1, 2], pads=[1, 0, 2, 0, 1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

start_time = time.time()
deconv_op.strides = [2, 1, 2]
deconv_op.pads = [1, 0, 2, 0, 1, 1]
deconv_op.dilations = None
deconv_op.auto_pad = "NOTSET"
deconv_op.output_padding = None
kp_out = deconv_op.run(numpy_in, numpy_w, numpy_b)[0]
print(f"{deconv_op}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 17: 1D ConvTranspose, group=2 basic
print("Case 17: 1D ConvTranspose, group=2 basic")
numpy_in = np.random.uniform(-1, 1, (2, 6, 16)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (6, 4, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, group=2, pads=[1, 1])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

deconv_op_g = ConvTransposeOp(mgr)
deconv_op_g.group = 2
deconv_op_g.auto_pad = "NOTSET"
deconv_op_g.kernel_shape = None
deconv_op_g.strides = None
deconv_op_g.pads = [1, 1]
deconv_op_g.dilations = None
deconv_op_g.output_padding = None
deconv_op_g.output_shape = None
start_time = time.time()
kp_out = deconv_op_g.run(numpy_in, numpy_w)[0]
print(f"{deconv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 18: 2D ConvTranspose, group=2 stride+pad
print("Case 18: 2D ConvTranspose, group=2 stride+pad")
numpy_in = np.random.uniform(-1, 1, (2, 6, 13, 11)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (6, 4, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, group=2, strides=[2, 1], pads=[1, 1, 2, 0])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

deconv_op_g = ConvTransposeOp(mgr)
deconv_op_g.group = 2
deconv_op_g.auto_pad = "NOTSET"
deconv_op_g.kernel_shape = None
deconv_op_g.strides = [2, 1]
deconv_op_g.pads = [1, 1, 2, 0]
deconv_op_g.dilations = None
deconv_op_g.output_padding = None
deconv_op_g.output_shape = None
start_time = time.time()
kp_out = deconv_op_g.run(numpy_in, numpy_w)[0]
print(f"{deconv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 19: 2D ConvTranspose, group=4 with dilation
print("Case 19: 2D ConvTranspose, group=4 with dilation")
numpy_in = np.random.uniform(-1, 1, (1, 8, 14, 15)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, group=4, dilations=[2, 1], strides=[1, 2])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

deconv_op_g = ConvTransposeOp(mgr)
deconv_op_g.group = 4
deconv_op_g.auto_pad = "NOTSET"
deconv_op_g.kernel_shape = None
deconv_op_g.strides = [1, 2]
deconv_op_g.pads = None
deconv_op_g.dilations = [2, 1]
deconv_op_g.output_padding = None
deconv_op_g.output_shape = None
start_time = time.time()
kp_out = deconv_op_g.run(numpy_in, numpy_w)[0]
print(f"{deconv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()

# Case 20: 3D ConvTranspose, group=2 bias+pad+stride
print("Case 20: 3D ConvTranspose, group=2 bias+pad+stride")
numpy_in = np.random.uniform(-1, 1, (1, 4, 8, 9, 7)).astype(np.float32)
numpy_w = np.random.uniform(-1, 1, (4, 3, 3, 3, 2)).astype(np.float32)
numpy_b = np.random.uniform(-1, 1, (6,)).astype(np.float32)
start_time = time.time()
numpy_out = onnx_reference_conv_transpose(numpy_in, numpy_w, numpy_b, group=2, strides=[2, 1, 1], pads=[1, 0, 2, 0, 1, 0])
print("NumPy:", numpy_out.shape, time.time() - start_time, "seconds")

deconv_op_g = ConvTransposeOp(mgr)
deconv_op_g.group = 2
deconv_op_g.auto_pad = "NOTSET"
deconv_op_g.kernel_shape = None
deconv_op_g.strides = [2, 1, 1]
deconv_op_g.pads = [1, 0, 2, 0, 1, 0]
deconv_op_g.dilations = None
deconv_op_g.output_padding = None
deconv_op_g.output_shape = None
start_time = time.time()
kp_out = deconv_op_g.run(numpy_in, numpy_w, numpy_b)[0]
print(f"{deconv_op_g}:", kp_out.shape, time.time() - start_time, "seconds")
print("Max error:", np.abs(numpy_out - kp_out).max())
print("All close:", np.allclose(numpy_out, kp_out, rtol=1e-4, atol=1e-4))
print()
