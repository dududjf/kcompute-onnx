import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class ConvTransposeOp:

    def __init__(self, manager: kp.Manager, auto_pad="NOTSET", dilations=None, group=1,
                 kernel_shape=None, pads=None, strides=None, output_padding=None, output_shape=None):
        self.auto_pad = auto_pad
        self.dilations = dilations
        self.group = group
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.output_padding = output_padding
        self.output_shape = output_shape
        self.manager = manager
        self.deconv1d_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint n_id   = gl_GlobalInvocationID.x;
    uint ocg_id = gl_GlobalInvocationID.y;
    uint g_id   = gl_GlobalInvocationID.z;

    uint n        = params[0];
    uint cin      = params[1];
    uint cout     = params[2];
    uint i0n      = params[3];
    uint o0n      = params[4];
    uint k0       = params[5];
    uint s0       = params[6];
    uint p0       = params[7];
    uint d0       = params[8];
    uint in_stride_n  = params[9];
    uint in_stride_c  = params[10];
    uint in_stride_i0 = params[11];
    uint out_stride_n  = params[12];
    uint out_stride_c  = params[13];
    uint out_stride_o0 = params[14];
    uint has_bias = params[15];
    uint group     = params[16];
    uint cinpg     = params[17];
    uint coutpg    = params[18];
    uint w_stride_w0 = params[19];
    uint w_stride_w1 = params[20];
    uint w_stride_k0 = params[21];

    if(n_id >= n || ocg_id >= coutpg || g_id >= group) return;

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint out_base_n = n_id * out_stride_n + oc_tot * out_stride_c;
    const float bias_val = (has_bias == 1u) ? b_buf[oc_tot] : 0.0;

    for (uint o0i = 0; o0i < o0n; ++o0i) {{
        const uint out_idx = out_base_n + o0i * out_stride_o0;
        y_buf[out_idx] = bias_val;
    }}

    for (uint ci = 0; ci < cinpg; ++ci) {{
        const uint cin_tot = g_id * cinpg + ci;
        const uint w_base = cin_tot * w_stride_w0 + ocg_id * w_stride_w1;
        const uint in_base_c = n_id * in_stride_n + cin_tot * in_stride_c;

        for (uint i0 = 0; i0 < i0n; ++i0) {{
            const uint in_idx = in_base_c + i0 * in_stride_i0;
            const float x_val = x_buf[in_idx];

            for (uint kk = 0; kk < k0; ++kk) {{
                const uint w_idx = w_base + kk * w_stride_k0;
                const float w_val = w_buf[w_idx];

                const int t = int(i0) * int(s0);
                const int o0i = t - int(p0) + int(kk) * int(d0);

                if (o0i >= 0 && uint(o0i) < o0n) {{
                    const uint out_idx = out_base_n + uint(o0i) * out_stride_o0;
                    // Atomic add not required if output pixel processed by unique thread?
                    // ConvTranspose: Multiple inputs contribute to same output.
                    // THIS SHADER IS A SCATTER PATTERN (Input driven)!
                    // Original implementation use naive scattering which requires Atomic Add for floats?
                    // GLSL 450 doesn't support atomicAdd for float directly.
                    // Wait, the original code used `y_buf[out_idx] += val`.
                    // Is `y_buf` shared? Yes!
                    // Is `gl_GlobalInvocationID` mapping to unique output? NO.
                    // Thread maps to (n_id, ocg_id, g_id) -> ONE output channel of ONE sample.
                    // Inside the thread, it iterates over ALL input pixels (i0) and kernel (kk).
                    // So one thread computes Contributions from ALL Inputs to ALL Outputs for that Channel.
                    // So there is NO race condition between threads for `y_buf[out_idx]`.
                    // Each thread owns `out_base_n` slice (one channel).
                    // So `+=` is safe.
                    y_buf[out_idx] += x_val * w_val;
                }}
            }}
        }}
    }}
}}
""")
        self.deconv2d_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint n_id   = gl_GlobalInvocationID.x;
    uint ocg_id = gl_GlobalInvocationID.y;
    uint g_id   = gl_GlobalInvocationID.z;

    uint n        = params[0];
    uint cin      = params[1];
    uint cout     = params[2];
    uint i0n      = params[3];
    uint i1n      = params[4];
    uint o0n      = params[5];
    uint o1n      = params[6];
    uint k0       = params[7];
    uint k1       = params[8];
    uint s0       = params[9];
    uint s1       = params[10];
    uint p0       = params[11];
    uint p1       = params[12];
    uint d0       = params[13];
    uint d1       = params[14];
    uint in_stride_n  = params[15];
    uint in_stride_c  = params[16];
    uint in_stride_i0 = params[17];
    uint in_stride_i1 = params[18];
    uint out_stride_n  = params[19];
    uint out_stride_c  = params[20];
    uint out_stride_o0 = params[21];
    uint out_stride_o1 = params[22];
    uint has_bias = params[23];
    uint group     = params[24];
    uint cinpg     = params[25];
    uint coutpg    = params[26];
    uint w_stride_w0 = params[27];
    uint w_stride_w1 = params[28];
    uint w_stride_k0 = params[29];
    uint w_stride_k1 = params[30];

    if(n_id >= n || ocg_id >= coutpg || g_id >= group) return;

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint out_base_n = n_id * out_stride_n + oc_tot * out_stride_c;
    const float bias_val = (has_bias == 1u) ? b_buf[oc_tot] : 0.0;

    for (uint o0i = 0; o0i < o0n; ++o0i) {{
        const uint out_base_o0 = out_base_n + o0i * out_stride_o0;
        for (uint o1i = 0; o1i < o1n; ++o1i) {{
            const uint out_idx = out_base_o0 + o1i * out_stride_o1;
            y_buf[out_idx] = bias_val;
        }}
    }}

    for (uint ci = 0; ci < cinpg; ++ci) {{
        const uint cin_tot = g_id * cinpg + ci;
        const uint w_base = cin_tot * w_stride_w0 + ocg_id * w_stride_w1;
        const uint in_base_c = n_id * in_stride_n + cin_tot * in_stride_c;

        for (uint i0 = 0; i0 < i0n; ++i0) {{
            const uint in_base_i0 = in_base_c + i0 * in_stride_i0;
            for (uint i1 = 0; i1 < i1n; ++i1) {{
                const uint in_idx = in_base_i0 + i1 * in_stride_i1;
                const float x_val = x_buf[in_idx];

                for (uint k0i = 0; k0i < k0; ++k0i) {{
                    const uint w_k0 = w_base + k0i * w_stride_k0;
                    for (uint k1i = 0; k1i < k1; ++k1i) {{
                        const uint w_idx = w_k0 + k1i * w_stride_k1;
                        const float w_val = w_buf[w_idx];

                        const int t0 = int(i0) * int(s0);
                        const int t1 = int(i1) * int(s1);
                        const int o0i = t0 - int(p0) + int(k0i) * int(d0);
                        const int o1i = t1 - int(p1) + int(k1i) * int(d1);

                        if (o0i >= 0 && uint(o0i) < o0n && o1i >= 0 && uint(o1i) < o1n) {{
                            const uint out_idx = out_base_n + uint(o0i) * out_stride_o0 + uint(o1i) * out_stride_o1;
                            y_buf[out_idx] += x_val * w_val;
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
""")
        self.deconv3d_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint n_id   = gl_GlobalInvocationID.x;
    uint ocg_id = gl_GlobalInvocationID.y;
    uint g_id   = gl_GlobalInvocationID.z;

    uint n        = params[0];
    uint cin      = params[1];
    uint cout     = params[2];
    uint i0n      = params[3];
    uint i1n      = params[4];
    uint i2n      = params[5];
    uint o0n      = params[6];
    uint o1n      = params[7];
    uint o2n      = params[8];
    uint k0       = params[9];
    uint k1       = params[10];
    uint k2       = params[11];
    uint s0       = params[12];
    uint s1       = params[13];
    uint s2       = params[14];
    uint p0       = params[15];
    uint p1       = params[16];
    uint p2       = params[17];
    uint d0       = params[18];
    uint d1       = params[19];
    uint d2       = params[20];
    uint in_stride_n  = params[21];
    uint in_stride_c  = params[22];
    uint in_stride_i0 = params[23];
    uint in_stride_i1 = params[24];
    uint in_stride_i2 = params[25];
    uint out_stride_n  = params[26];
    uint out_stride_c  = params[27];
    uint out_stride_o0 = params[28];
    uint out_stride_o1 = params[29];
    uint out_stride_o2 = params[30];
    uint has_bias = params[31];
    uint group     = params[32];
    uint cinpg     = params[33];
    uint coutpg    = params[34];
    uint w_stride_w0 = params[35];
    uint w_stride_w1 = params[36];
    uint w_stride_k0 = params[37];
    uint w_stride_k1 = params[38];
    uint w_stride_k2 = params[39];

    if(n_id >= n || ocg_id >= coutpg || g_id >= group) return;

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint out_base_n = n_id * out_stride_n + oc_tot * out_stride_c;
    const float bias_val = (has_bias == 1u) ? b_buf[oc_tot] : 0.0;

    for (uint o0i = 0; o0i < o0n; ++o0i) {{
        const uint out_base_o0 = out_base_n + o0i * out_stride_o0;
        for (uint o1i = 0; o1i < o1n; ++o1i) {{
            const uint out_base_o1 = out_base_o0 + o1i * out_stride_o1;
            for (uint o2i = 0; o2i < o2n; ++o2i) {{
                const uint out_idx = out_base_o1 + o2i * out_stride_o2;
                y_buf[out_idx] = bias_val;
            }}
        }}
    }}

    for (uint ci = 0; ci < cinpg; ++ci) {{
        const uint cin_tot = g_id * cinpg + ci;
        const uint w_base = cin_tot * w_stride_w0 + ocg_id * w_stride_w1;
        const uint in_base_c = n_id * in_stride_n + cin_tot * in_stride_c;

        for (uint i0 = 0; i0 < i0n; ++i0) {{
            const uint in_base_i0 = in_base_c + i0 * in_stride_i0;
            for (uint i1 = 0; i1 < i1n; ++i1) {{
                const uint in_base_i1 = in_base_i0 + i1 * in_stride_i1;
                for (uint i2 = 0; i2 < i2n; ++i2) {{
                    const uint in_idx = in_base_i1 + i2 * in_stride_i2;
                    const float x_val = x_buf[in_idx];

                    for (uint k0i = 0; k0i < k0; ++k0i) {{
                        const uint w_k0 = w_base + k0i * w_stride_k0;
                        for (uint k1i = 0; k1i < k1; ++k1i) {{
                            const uint w_k1 = w_k0 + k1i * w_stride_k1;
                            for (uint k2i = 0; k2i < k2; ++k2i) {{
                                const uint w_idx = w_k1 + k2i * w_stride_k2;
                                const float w_val = w_buf[w_idx];

                                const int t0 = int(i0) * int(s0);
                                const int t1 = int(i1) * int(s1);
                                const int t2 = int(i2) * int(s2);
                                const int o0i = t0 - int(p0) + int(k0i) * int(d0);
                                const int o1i = t1 - int(p1) + int(k1i) * int(d1);
                                const int o2i = t2 - int(p2) + int(k2i) * int(d2);

                                if (o0i >= 0 && uint(o0i) < o0n && 
                                    o1i >= 0 && uint(o1i) < o1n && 
                                    o2i >= 0 && uint(o2i) < o2n) {{
                                    const uint out_idx = out_base_n + 
                                        uint(o0i) * out_stride_o0 + 
                                        uint(o1i) * out_stride_o1 + 
                                        uint(o2i) * out_stride_o2;
                                    y_buf[out_idx] += x_val * w_val;
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ConvTransposeOp({dev})"

    __str__ = __repr__

    # 输出形状与 pads 推导
    def _infer_output_shape_and_pads(self, X_spatial, W_spatial, auto_pad, dilations, strides,
                                     pads, output_padding, output_shape):
        dims = len(X_spatial)
        dils = dilations if dilations is not None else [1] * dims
        strs = strides if strides is not None else [1] * dims
        kshape = list(W_spatial)
        out_pad = output_padding if output_padding is not None else [0] * dims

        if pads is None and auto_pad not in {"SAME_UPPER", "SAME_LOWER"}:
            pads = [0 for _ in range(2 * dims)]

        if pads is None:
            # SAME_*: 需要根据输出形状推导 pads
            out_shape = list(output_shape) if output_shape is not None else [X_spatial[i] * strs[i] for i in range(dims)]
            total_padding = [
                strs[i] * (X_spatial[i] - 1) + out_pad[i] + ((kshape[i] - 1) * dils[i] + 1) - out_shape[i]
                for i in range(dims)
            ]
            pads_1 = []
            pads_2 = []
            for i in range(dims):
                int_half = total_padding[i] // 2
                if auto_pad == "SAME_UPPER":
                    pads_1.append(int_half)
                    pads_2.append(total_padding[i] - int_half)
            pads_used = pads_1 + pads_2
            out_shape_used = out_shape
        else:
            n_dims = dims
            if output_shape is None:
                out_shape_used = [
                    strs[i] * (X_spatial[i] - 1) + out_pad[i] + ((kshape[i] - 1) * dils[i] + 1) - (pads[i] + pads[i + n_dims])
                    for i in range(n_dims)
                ]
            else:
                out_shape_used = list(output_shape)
            pads_used = list(pads)

        # 返回去掉未使用的 out_pad
        return out_shape_used, pads_used, dils, strs, kshape

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        # 解析输入：X, W, [B]
        assert len(input_tensors) >= 2, "ConvTranspose expects at least two inputs: X and W"
        tensor_x, shape_x = input_tensors[0]
        tensor_w, shape_w = input_tensors[1]
        tensor_b = None
        shape_b = []
        if len(input_tensors) >= 3:
            tensor_b, shape_b = input_tensors[2]

        # 基本形状检查
        assert len(shape_x) >= 3, f"ConvTranspose expects at least 3D input (N, C_in, spatial...), got {shape_x}"
        n, cin = shape_x[0], shape_x[1]
        in_spatial = shape_x[2:]
        spatial_dims = len(in_spatial)

        # W: [Cin, Cout/group, ...]
        g = self.group or 1
        assert g >= 1, "group must be >= 1"
        assert len(shape_w) == 2 + spatial_dims, f"W rank mismatch, got {shape_w}"
        assert shape_w[0] == cin, f"W.shape[0] must be Cin ({cin}), got {shape_w[0]}"
        coutpg = shape_w[1]
        assert cin % g == 0, f"Cin {cin} must be divisible by group {g}"
        cinpg = cin // g
        cout = coutpg * g

        kernel = shape_w[2:] if self.kernel_shape is None else self.kernel_shape
        assert len(kernel) == spatial_dims

        # Bias
        has_bias = 0
        if tensor_b is not None:
            assert len(shape_b) == 1 and shape_b[0] == cout, f"Bias must be shape [Cout], got {shape_b}"
            has_bias = 1
        else:
            tensor_b = self.manager.tensor(np.zeros(1, dtype=np.float32))
            updated_tensors.append(tensor_b)

        # 属性
        auto_pad = self.auto_pad or "NOTSET"
        out_shape_attr = self.output_shape
        out_pad_attr = self.output_padding
        pads_in = self.pads
        strides_in = self.strides
        dilations_in = self.dilations

        # 输出形状 + pads（对齐ONNX参考）
        out_spatial, pads_final, dilations, strides, kernel_shape = self._infer_output_shape_and_pads(
            in_spatial, kernel, auto_pad, dilations_in, strides_in, pads_in, out_pad_attr, out_shape_attr
        )

        # 展开到3维
        i0 = in_spatial[0] if spatial_dims >= 1 else 1
        i1 = in_spatial[1] if spatial_dims >= 2 else 1
        i2 = in_spatial[2] if spatial_dims >= 3 else 1
        o0 = out_spatial[0] if spatial_dims >= 1 else 1
        o1 = out_spatial[1] if spatial_dims >= 2 else 1
        o2 = out_spatial[2] if spatial_dims >= 3 else 1
        k0 = kernel_shape[0] if spatial_dims >= 1 else 1
        k1 = kernel_shape[1] if spatial_dims >= 2 else 1
        k2 = kernel_shape[2] if spatial_dims >= 3 else 1
        s0 = strides[0] if spatial_dims >= 1 else 1
        s1 = strides[1] if spatial_dims >= 2 else 1
        s2 = strides[2] if spatial_dims >= 3 else 1
        p0 = pads_final[0] if spatial_dims >= 1 else 0
        p1 = pads_final[1] if spatial_dims >= 2 else 0
        p2 = pads_final[2] if spatial_dims >= 3 else 0
        d0 = dilations[0] if spatial_dims >= 1 else 1
        d1 = dilations[1] if spatial_dims >= 2 else 1
        d2 = dilations[2] if spatial_dims >= 3 else 1

        # 输入步长（flatten）
        in_stride_i2 = 1
        in_stride_i1 = i2
        in_stride_i0 = i1 * i2
        in_stride_c  = i0 * i1 * i2
        in_stride_n  = cin * in_stride_c

        # 输出步长（flatten）
        out_stride_o2 = 1
        out_stride_o1 = o2
        out_stride_o0 = o1 * o2
        out_stride_c  = o0 * o1 * o2
        out_stride_n  = cout * out_stride_c

        # 权值步长（W: [Cin, Cout/group, K...], row-major）
        w_stride_k2 = 1
        w_stride_k1 = k2
        w_stride_k0 = k1 * k2
        w_stride_w1 = k0 * k1 * k2
        w_stride_w0 = coutpg * w_stride_w1

        # 输出 tensor
        shape_out = [n, cout] + out_spatial
        tensor_out = self.manager.tensor(np.zeros(int(np.prod(shape_out)), dtype=np.float32))
        updated_tensors.append(tensor_out)

        # 调度维度
        group_x = (n + LOCAL_X_3D - 1) // LOCAL_X_3D
        group_y = (coutpg + LOCAL_Y_3D - 1) // LOCAL_Y_3D
        group_z = (g + LOCAL_Z_3D - 1) // LOCAL_Z_3D

        if spatial_dims == 1:
            params = np.array([
                n, cin, cout,
                i0, o0, k0, s0, p0, d0,
                in_stride_n, in_stride_c, in_stride_i0,
                out_stride_n, out_stride_c, out_stride_o0,
                has_bias, g, cinpg, coutpg,
                w_stride_w0, w_stride_w1, w_stride_k0,
            ], dtype=np.uint32)
            shader = self.deconv1d_shader
        elif spatial_dims == 2:
            params = np.array([
                n, cin, cout, i0, i1, o0, o1, k0, k1,
                s0, s1, p0, p1, d0, d1,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1,
                has_bias, g, cinpg, coutpg,
                w_stride_w0, w_stride_w1, w_stride_k0, w_stride_k1,
            ], dtype=np.uint32)
            shader = self.deconv2d_shader
        else:
            params = np.array([
                n, cin, cout, i0, i1, i2, o0, o1, o2, k0, k1, k2,
                s0, s1, s2, p0, p1, p2, d0, d1, d2,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1, in_stride_i2,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1, out_stride_o2,
                has_bias, g, cinpg, coutpg,
                w_stride_w0, w_stride_w1, w_stride_k0, w_stride_k1, w_stride_k2,
            ], dtype=np.uint32)
            shader = self.deconv3d_shader

        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, tensor_w, tensor_b, tensor_out, param_in],
            shader,
            (group_x, group_y, group_z)
        ))

        return [(tensor_out, shape_out)]

