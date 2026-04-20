import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class ConvOp:

    def __init__(self, manager: kp.Manager, auto_pad="NOTSET", dilations=None, group=1,
                 kernel_shape=None, pads=None, strides=None):
        self.auto_pad = auto_pad
        self.dilations = dilations
        self.group = group
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.manager = manager
        self.conv1d_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }}; // 输入特征图X
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }}; // 卷积核W
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }}; // 偏置B（可选）
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }}; // 输出特征图Y
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint n_id  = gl_GlobalInvocationID.x;   // batch索引
    uint ocg_id = gl_GlobalInvocationID.y;  // 组内输出通道索引
    uint g_id  = gl_GlobalInvocationID.z;   // 组索引

    uint n        = params[0];
    uint cin      = params[1]; // Unused but passed
    uint cout     = params[2]; // Unused but passed
    uint i0       = params[3];
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
    uint inc_i0    = params[15];
    uint has_bias  = params[16];
    uint group     = params[17];
    uint cinpg     = params[18];
    uint coutpg    = params[19];

    if(n_id >= n || ocg_id >= coutpg || g_id >= group) return;

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint w_base_oc = oc_tot * cinpg * k0;
    const uint in_base_n = n_id * in_stride_n;
    const uint out_base = n_id * out_stride_n + oc_tot * out_stride_c;

    for (uint o0i = 0; o0i < o0n; ++o0i) {{
        const uint out_idx = out_base + o0i * out_stride_o0;

        const int i0s = int(o0i * s0) - int(p0);

        float acc = (has_bias == 1u) ? b_buf[oc_tot] : 0.0;

        for (uint ci = 0; ci < cinpg; ++ci) {{
            const uint in_base_ci = in_base_n + (g_id * cinpg + ci) * in_stride_c;
            uint in_i = in_base_ci + uint(i0s) * in_stride_i0;

            const uint w_base_ci = w_base_oc + ci * k0;
            uint w_k = w_base_ci;

            int cur_i = i0s;
            for (uint kk = 0; kk < k0; ++kk) {{
                if (cur_i >= 0 && cur_i < int(i0)) {{
                    acc += x_buf[in_i] * w_buf[w_k];
                }}
                cur_i += int(d0);
                in_i  += inc_i0;
                w_k   += 1;
            }}
        }}

        y_buf[out_idx] = acc;
    }}
}}
""")
        self.conv2d_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }}; // 输入特征图X
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }}; // 卷积核W
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }}; // 偏置B（可选）
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }}; // 输出特征图Y
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint n_id  = gl_GlobalInvocationID.x;   // batch索引
    uint ocg_id = gl_GlobalInvocationID.y;  // 组内输出通道索引
    uint g_id  = gl_GlobalInvocationID.z;   // 组索引

    uint n        = params[0];
    uint cin      = params[1];
    uint cout     = params[2];
    uint i0       = params[3];
    uint i1       = params[4];
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
    uint inc_i0    = params[23];
    uint inc_i1    = params[24];
    uint has_bias  = params[25];
    uint group     = params[26];
    uint cinpg     = params[27];
    uint coutpg    = params[28];
    
    if(n_id >= n || ocg_id >= coutpg || g_id >= group) return;

    const uint k0k1      = k0 * k1;
    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint w_base_oc = oc_tot * cinpg * k0k1;
    const uint in_base_n = n_id * in_stride_n;
    const uint in_base_g = in_base_n + g_id * cinpg * in_stride_c;
    const uint out_base = n_id * out_stride_n + oc_tot * out_stride_c;

    for (uint o0i = 0; o0i < o0n; ++o0i) {{
        const uint out_base_o0 = out_base + o0i * out_stride_o0;

        const int i0s = int(o0i * s0) - int(p0);

        for (uint o1i = 0; o1i < o1n; ++o1i) {{
            const uint out_idx = out_base_o0 + o1i * out_stride_o1;

            const int i1s = int(o1i * s1) - int(p1);

            float acc = (has_bias == 1u) ? b_buf[oc_tot] : 0.0;

            for (uint ci = 0; ci < cinpg; ++ci) {{
                const uint in_base_ci = in_base_g + ci * in_stride_c;
                uint in_i0 = in_base_ci + uint(i0s) * in_stride_i0;

                const uint w_base_ci = w_base_oc + ci * k0k1;
                uint w_k0 = w_base_ci;

                int cur_i0 = i0s;
                for (uint k0i = 0; k0i < k0; ++k0i) {{
                    if (cur_i0 >= 0 && cur_i0 < int(i0)) {{
                        uint in_i1 = in_i0 + uint(i1s) * in_stride_i1;
                        uint w_k1 = w_k0;

                        int cur_i1 = i1s;
                        for (uint k1i = 0; k1i < k1; ++k1i) {{
                            if (cur_i1 >= 0 && cur_i1 < int(i1)) {{
                                acc += x_buf[in_i1] * w_buf[w_k1];
                            }}
                            cur_i1 += int(d1);
                            in_i1  += inc_i1;
                            w_k1   += 1;
                        }}
                    }}
                    cur_i0 += int(d0);
                    in_i0  += inc_i0;
                    w_k0   += k1;
                }}
            }}

            y_buf[out_idx] = acc;
        }}
    }}
}}
""")
        self.conv3d_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }}; // 输入特征图X
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }}; // 卷积核W
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }}; // 偏置B（可选）
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }}; // 输出特征图Y
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint n_id  = gl_GlobalInvocationID.x;   // batch索引
    uint ocg_id = gl_GlobalInvocationID.y;  // 组内输出通道索引
    uint g_id  = gl_GlobalInvocationID.z;   // 组索引

    uint n        = params[0];
    uint cin      = params[1];
    uint cout     = params[2];
    uint i0       = params[3];
    uint i1       = params[4];
    uint i2       = params[5];
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
    uint k1k2     = params[31];
    uint k0k1k2   = params[32];
    uint inc_i0   = params[33];
    uint inc_i1   = params[34];
    uint inc_i2   = params[35];
    uint has_bias = params[36];
    uint group    = params[37];
    uint cinpg    = params[38];
    uint coutpg   = params[39];

    if(n_id >= n || ocg_id >= coutpg || g_id >= group) return;

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint w_base_oc = oc_tot * cinpg * k0k1k2;
    const uint in_base_n = n_id * in_stride_n;
    const uint in_base_g = in_base_n + g_id * cinpg * in_stride_c;
    const uint out_base = n_id * out_stride_n + oc_tot * out_stride_c;

    for (uint o0i = 0; o0i < o0n; ++o0i) {{
        const uint out_base_o0 = out_base + o0i * out_stride_o0;

        const int i0s = int(o0i * s0) - int(p0);

        for (uint o1i = 0; o1i < o1n; ++o1i) {{
            const uint out_base_o1 = out_base_o0 + o1i * out_stride_o1;

            const int i1s = int(o1i * s1) - int(p1);

            for (uint o2i = 0; o2i < o2n; ++o2i) {{
                const uint out_idx = out_base_o1 + o2i * out_stride_o2;

                const int i2s = int(o2i * s2) - int(p2);

                float acc = (has_bias == 1u) ? b_buf[oc_tot] : 0.0;

                for (uint ci = 0; ci < cinpg; ++ci) {{
                    const uint in_base_ci = in_base_g + ci * in_stride_c;
                    uint in_i0 = in_base_ci + uint(i0s) * in_stride_i0;
                    const uint w_base_ci = w_base_oc + ci * k0k1k2;
                    uint w_k0 = w_base_ci;

                    int cur_i0 = i0s;
                    for (uint k0i = 0; k0i < k0; ++k0i) {{
                        if (cur_i0 >= 0 && cur_i0 < int(i0)) {{
                            uint in_i1 = in_i0 + uint(i1s) * in_stride_i1;
                            uint w_k1 = w_k0;

                            int cur_i1 = i1s;
                            for (uint k1i = 0; k1i < k1; ++k1i) {{
                                if (cur_i1 >= 0 && cur_i1 < int(i1)) {{
                                    uint in_i2 = in_i1 + uint(i2s) * in_stride_i2;
                                    uint w_k2 = w_k1;

                                    int cur_i2 = i2s; 
                                    for (uint k2i = 0; k2i < k2; ++k2i) {{
                                        if (cur_i2 >= 0 && cur_i2 < int(i2)) {{
                                            acc += x_buf[in_i2] * w_buf[w_k2];
                                        }}

                                        cur_i2 += int(d2);
                                        in_i2  += inc_i2;
                                        w_k2   += 1;
                                    }}
                                }}
                                cur_i1 += int(d1);
                                in_i1  += inc_i1;
                                w_k1   += k2;
                            }}
                        }}
                        cur_i0 += int(d0);
                        in_i0  += inc_i0;
                        w_k0   += k1k2;
                    }}
                }}

                y_buf[out_idx] = acc;
            }}
        }}
    }}
}}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ConvOp({dev})"

    __str__ = __repr__

    def _get_output_shape_explicit_padding(self, input_spatial_shape, kernel_spatial_shape, strides_spatial, pads,
                                           dilations):
        dims = len(input_spatial_shape)
        dilations = dilations or [1] * dims
        out_shape = [0] * dims
        pad_list = list(pads or [0] * (2 * dims))

        for dim in range(dims):
            eff_k = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
            num = (input_spatial_shape[dim] + pad_list[dim] + pad_list[dims + dim]
                   - eff_k) / strides_spatial[dim] + 1.0
            out_shape[dim] = int(np.floor(num))

        pads_new = pad_list[:]
        return out_shape, pads_new

    def _get_output_shape_auto_pad(self, auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, dilations):
        out_shape = [0] * len(input_spatial_shape)
        for i in range(len(input_spatial_shape)):
            if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
                out_shape[i] = int(np.floor((input_spatial_shape[i] - 1) / strides_spatial[i])) + 1
            elif auto_pad == "VALID":
                eff_k = (kernel_spatial_shape[i] - 1) * dilations[i] + 1
                out_shape[i] = int(np.floor((input_spatial_shape[i] - eff_k) / strides_spatial[i])) + 1
        return out_shape

    def _get_pad_shape(self, auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, out_shape, dilations):
        pad_shape = [0] * len(input_spatial_shape)
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            for i in range(len(input_spatial_shape)):
                eff_k = (kernel_spatial_shape[i] - 1) * dilations[i] + 1
                pad_shape[i] = (out_shape[i] - 1) * strides_spatial[i] + eff_k - input_spatial_shape[i]
        return pad_shape

    def _get_pad_with_auto_pad(self, auto_pad, pad_shape):
        dims = len(pad_shape)
        if auto_pad == "SAME_UPPER":
            return [pad_shape[i] // 2 for i in range(dims)] + [pad_shape[i] - pad_shape[i] // 2 for i in range(dims)]
        elif auto_pad == "SAME_LOWER":
            return [pad_shape[i] - pad_shape[i] // 2 for i in range(dims)] + [pad_shape[i] // 2 for i in range(dims)]
        else:
            return [0] * dims * 2

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
        assert len(input_tensors) >= 2, "Conv expects at least two inputs: X and W"
        tensor_x, shape_x = input_tensors[0]
        tensor_w, shape_w = input_tensors[1]
        tensor_b = None
        shape_b = []
        if len(input_tensors) >= 3:
            tensor_b, shape_b = input_tensors[2]

        # 基本形状检查
        assert len(shape_x) >= 3, f"Conv expects at least 3D input (N, C_in, spatial...), got {shape_x}"
        n, cin = shape_x[0], shape_x[1]
        in_spatial = shape_x[2:]
        spatial_dims = len(in_spatial)

        # 组卷积支持：W 形状 [Cout, Cin/group, ...]
        g = self.group or 1
        assert g >= 1, "group must be >= 1"
        assert len(shape_w) == 2 + spatial_dims, f"W rank mismatch, got {shape_w}"
        cout = shape_w[0]
        cinpg_expected = shape_w[1]
        assert cin % g == 0, f"Cin {cin} must be divisible by group {g}"
        assert cout % g == 0, f"Cout {cout} must be divisible by group {g}"
        cinpg = cin // g
        coutpg = cout // g
        assert cinpg_expected == cinpg, f"W.shape[1] must be Cin/group ({cinpg}), got {cinpg_expected}"

        kernel = shape_w[2:] if self.kernel_shape is None else self.kernel_shape
        assert len(kernel) == spatial_dims

        # 可选 Bias
        has_bias = 0
        if tensor_b is not None:
            assert len(shape_b) == 1 and shape_b[0] == cout, f"Bias must be shape [Cout], got {shape_b}"
            has_bias = 1
        else:
            # Need a dummy buffer for bias because shader expects it binding=2
            tensor_b = self.manager.tensor(np.array([0.0], dtype=np.float32))
            # Sync dummy bias
            self.manager.sequence().record(kp.OpTensorSyncDevice([tensor_b])).eval()

        # 属性
        strides = self.strides if self.strides is not None else [1] * spatial_dims
        dilations = self.dilations if self.dilations is not None else [1] * spatial_dims
        dilations = [d if d > 0 else 1 for d in dilations]
        pads_in = self.pads if self.pads is not None else [0] * (2 * spatial_dims)
        auto_pad = self.auto_pad or "NOTSET"

        # 输出形状与pad
        if auto_pad in ("SAME_UPPER", "SAME_LOWER", "VALID"):
            assert self.pads is None, f"pads cannot be specified when auto_pad is not 'NOTSET'. Got auto_pad='{auto_pad}' and pads={self.pads}"
            out_spatial = self._get_output_shape_auto_pad(auto_pad, in_spatial, kernel, strides, dilations)
            pad_shape = self._get_pad_shape(auto_pad, in_spatial, kernel, strides, out_spatial, dilations) if auto_pad != "VALID" else [0] * spatial_dims
            pads_final = self._get_pad_with_auto_pad(auto_pad, pad_shape)
        else:
            out_spatial, pads_final = self._get_output_shape_explicit_padding(in_spatial, kernel, strides, pads_in, dilations)

        # 预计算展开尺寸
        i0 = in_spatial[0] if spatial_dims >= 1 else 1
        i1 = in_spatial[1] if spatial_dims >= 2 else 1
        i2 = in_spatial[2] if spatial_dims >= 3 else 1
        o0 = out_spatial[0] if spatial_dims >= 1 else 1
        o1 = out_spatial[1] if spatial_dims >= 2 else 1
        o2 = out_spatial[2] if spatial_dims >= 3 else 1
        k0 = kernel[0] if spatial_dims >= 1 else 1
        k1 = kernel[1] if spatial_dims >= 2 else 1
        k2 = kernel[2] if spatial_dims >= 3 else 1
        s0 = strides[0] if spatial_dims >= 1 else 1
        s1 = strides[1] if spatial_dims >= 2 else 1
        s2 = strides[2] if spatial_dims >= 3 else 1
        p0 = pads_final[0] if spatial_dims >= 1 else 0
        p1 = pads_final[1] if spatial_dims >= 2 else 0
        p2 = pads_final[2] if spatial_dims >= 3 else 0
        d0 = dilations[0] if spatial_dims >= 1 else 1
        d1 = dilations[1] if spatial_dims >= 2 else 1
        d2 = dilations[2] if spatial_dims >= 3 else 1

        # 步长（flatten）
        in_stride_i2 = 1
        in_stride_i1 = i2
        in_stride_i0 = i1 * i2
        in_stride_c  = i0 * i1 * i2
        in_stride_n  = cin * in_stride_c

        out_stride_o2 = 1
        out_stride_o1 = o2
        out_stride_o0 = o1 * o2
        out_stride_c  = o0 * o1 * o2
        out_stride_n  = cout * out_stride_c

        # 输出tensor
        shape_out = [n, cout] + out_spatial
        tensor_out = self.manager.tensor(np.zeros(int(np.prod(shape_out)), dtype=np.float32))
        updated_tensors.append(tensor_out)

        # 将(n, oc_in_group, group) 映射到 (x, y, z)
        # However, gl_GlobalInvocationID is (batch, coutpg, group)

        group_x = (n + LOCAL_X_3D - 1) // LOCAL_X_3D
        group_y = (coutpg + LOCAL_Y_3D - 1) // LOCAL_Y_3D
        group_z = (g + LOCAL_Z_3D - 1) // LOCAL_Z_3D

        if spatial_dims == 1:
            inc_i0 = d0 * in_stride_i0
            params = np.array([
                n, cin, cout, i0, o0, k0, s0, p0, d0,
                in_stride_n, in_stride_c, in_stride_i0,
                out_stride_n, out_stride_c, out_stride_o0,
                inc_i0, has_bias, g, cinpg, coutpg,
            ], dtype=np.uint32)
            shader = self.conv1d_shader
        elif spatial_dims == 2:
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            params = np.array([
                n, cin, cout, i0, i1, o0, o1, k0, k1, s0, s1, p0, p1, d0, d1,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1,
                inc_i0, inc_i1, has_bias, g, cinpg, coutpg,
            ], dtype=np.uint32)
            shader = self.conv2d_shader
        else:
            k1k2 = k1 * k2
            k0k1k2 = k0 * k1k2
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            inc_i2 = d2
            params = np.array([
                n, cin, cout, i0, i1, i2, o0, o1, o2, k0, k1, k2, s0, s1, s2, p0, p1, p2, d0, d1, d2,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1, in_stride_i2,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1, out_stride_o2,
                k1k2, k0k1k2, inc_i0, inc_i1, inc_i2, has_bias, g, cinpg, coutpg,
            ], dtype=np.uint32)
            shader = self.conv3d_shader

        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, tensor_w, tensor_b, tensor_out, param_in],
            shader,
            (group_x, group_y, group_z)
        ))

        return [(tensor_out, shape_out)]

