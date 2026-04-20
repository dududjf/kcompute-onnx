import kp
import numpy as np
from .shader_utils import compile_source


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
        self.deconv1d_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; };
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; };
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; };

layout(constant_id = 0)  const float n_f  = 0.0;
layout(constant_id = 1)  const float cin_f  = 0.0;
layout(constant_id = 2)  const float cout_f = 0.0;
layout(constant_id = 3)  const float i0_f = 1.0;
layout(constant_id = 4)  const float o0_f = 1.0;
layout(constant_id = 5)  const float k0_f = 1.0;
layout(constant_id = 6)  const float s0_f = 1.0;
layout(constant_id = 7)  const float p0_f = 0.0;
layout(constant_id = 8)  const float d0_f = 1.0;
layout(constant_id = 9)  const float in_stride_n_f  = 1.0;
layout(constant_id = 10) const float in_stride_c_f  = 1.0;
layout(constant_id = 11) const float in_stride_i0_f = 1.0;
layout(constant_id = 12) const float out_stride_n_f  = 1.0;
layout(constant_id = 13) const float out_stride_c_f  = 1.0;
layout(constant_id = 14) const float out_stride_o0_f = 1.0;
layout(constant_id = 15) const float has_bias_f = 0.0;
layout(constant_id = 16) const float group_f  = 1.0;
layout(constant_id = 17) const float cinpg_f  = 1.0;
layout(constant_id = 18) const float coutpg_f = 1.0;
layout(constant_id = 19) const float w_stride_w0_f = 1.0;
layout(constant_id = 20) const float w_stride_w1_f = 1.0;
layout(constant_id = 21) const float w_stride_k0_f = 1.0;

void main() {
    uint n_id   = gl_GlobalInvocationID.x;
    uint ocg_id = gl_GlobalInvocationID.y;
    uint g_id   = gl_GlobalInvocationID.z;

    const uint n        = uint(n_f);
    const uint i0n      = uint(i0_f);
    const uint o0n      = uint(o0_f);
    const uint k0       = uint(k0_f);
    const uint s0       = uint(s0_f);
    const uint p0       = uint(p0_f);
    const uint d0       = uint(d0_f);
    const uint in_stride_n  = uint(in_stride_n_f);
    const uint in_stride_c  = uint(in_stride_c_f);
    const uint in_stride_i0 = uint(in_stride_i0_f);
    const uint out_stride_n  = uint(out_stride_n_f);
    const uint out_stride_c  = uint(out_stride_c_f);
    const uint out_stride_o0 = uint(out_stride_o0_f);
    const uint has_bias = uint(has_bias_f);
    const uint group     = uint(group_f);
    const uint cinpg     = uint(cinpg_f);
    const uint coutpg    = uint(coutpg_f);
    const uint w_stride_w0 = uint(w_stride_w0_f);
    const uint w_stride_w1 = uint(w_stride_w1_f);
    const uint w_stride_k0 = uint(w_stride_k0_f);

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint out_base_n = n_id * out_stride_n + oc_tot * out_stride_c;
    const float bias_val = has_bias == 1 ? b_buf[oc_tot] : 0.0;

    for (uint o0i = 0; o0i < o0n; ++o0i) {
        const uint out_idx = out_base_n + o0i * out_stride_o0;
        y_buf[out_idx] = bias_val;
    }

    for (uint ci = 0; ci < cinpg; ++ci) {
        const uint cin_tot = g_id * cinpg + ci;
        const uint w_base = cin_tot * w_stride_w0 + ocg_id * w_stride_w1;
        const uint in_base_c = n_id * in_stride_n + cin_tot * in_stride_c;

        for (uint i0 = 0; i0 < i0n; ++i0) {
            const uint in_idx = in_base_c + i0 * in_stride_i0;
            const float x_val = x_buf[in_idx];

            for (uint kk = 0; kk < k0; ++kk) {
                const uint w_idx = w_base + kk * w_stride_k0;
                const float w_val = w_buf[w_idx];

                const int t = int(i0) * int(s0);
                const int o0i = t - int(p0) + int(kk) * int(d0);

                if (o0i >= 0 && uint(o0i) < o0n) {
                    const uint out_idx = out_base_n + uint(o0i) * out_stride_o0;
                    y_buf[out_idx] += x_val * w_val;
                }
            }
        }
    }
}
""")
        self.deconv2d_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; };
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; };
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; };

layout(constant_id = 0)  const float n_f  = 0.0;
layout(constant_id = 1)  const float cin_f  = 0.0;
layout(constant_id = 2)  const float cout_f = 0.0;
layout(constant_id = 3)  const float i0_f = 1.0;
layout(constant_id = 4)  const float i1_f = 1.0;
layout(constant_id = 5)  const float o0_f = 1.0;
layout(constant_id = 6)  const float o1_f = 1.0;
layout(constant_id = 7)  const float k0_f = 1.0;
layout(constant_id = 8)  const float k1_f = 1.0;
layout(constant_id = 9)  const float s0_f = 1.0;
layout(constant_id = 10) const float s1_f = 1.0;
layout(constant_id = 11) const float p0_f = 0.0;
layout(constant_id = 12) const float p1_f = 0.0;
layout(constant_id = 13) const float d0_f = 1.0;
layout(constant_id = 14) const float d1_f = 1.0;
layout(constant_id = 15) const float in_stride_n_f  = 1.0;
layout(constant_id = 16) const float in_stride_c_f  = 1.0;
layout(constant_id = 17) const float in_stride_i0_f = 1.0;
layout(constant_id = 18) const float in_stride_i1_f = 1.0;
layout(constant_id = 19) const float out_stride_n_f  = 1.0;
layout(constant_id = 20) const float out_stride_c_f  = 1.0;
layout(constant_id = 21) const float out_stride_o0_f = 1.0;
layout(constant_id = 22) const float out_stride_o1_f = 1.0;
layout(constant_id = 23) const float has_bias_f = 0.0;
layout(constant_id = 24) const float group_f  = 1.0;
layout(constant_id = 25) const float cinpg_f  = 1.0;
layout(constant_id = 26) const float coutpg_f = 1.0;
layout(constant_id = 27) const float w_stride_w0_f = 1.0;
layout(constant_id = 28) const float w_stride_w1_f = 1.0;
layout(constant_id = 29) const float w_stride_k0_f = 1.0;
layout(constant_id = 30) const float w_stride_k1_f = 1.0;

void main() {
    uint n_id   = gl_GlobalInvocationID.x;
    uint ocg_id = gl_GlobalInvocationID.y;
    uint g_id   = gl_GlobalInvocationID.z;

    const uint n        = uint(n_f);
    const uint i0n      = uint(i0_f);
    const uint i1n      = uint(i1_f);
    const uint o0n      = uint(o0_f);
    const uint o1n      = uint(o1_f);
    const uint k0       = uint(k0_f);
    const uint k1       = uint(k1_f);
    const uint s0       = uint(s0_f);
    const uint s1       = uint(s1_f);
    const uint p0       = uint(p0_f);
    const uint p1       = uint(p1_f);
    const uint d0       = uint(d0_f);
    const uint d1       = uint(d1_f);
    const uint in_stride_n  = uint(in_stride_n_f);
    const uint in_stride_c  = uint(in_stride_c_f);
    const uint in_stride_i0 = uint(in_stride_i0_f);
    const uint in_stride_i1 = uint(in_stride_i1_f);
    const uint out_stride_n  = uint(out_stride_n_f);
    const uint out_stride_c  = uint(out_stride_c_f);
    const uint out_stride_o0 = uint(out_stride_o0_f);
    const uint out_stride_o1 = uint(out_stride_o1_f);
    const uint has_bias = uint(has_bias_f);
    const uint group     = uint(group_f);
    const uint cinpg     = uint(cinpg_f);
    const uint coutpg    = uint(coutpg_f);
    const uint w_stride_w0 = uint(w_stride_w0_f);
    const uint w_stride_w1 = uint(w_stride_w1_f);
    const uint w_stride_k0 = uint(w_stride_k0_f);
    const uint w_stride_k1 = uint(w_stride_k1_f);

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint out_base_n = n_id * out_stride_n + oc_tot * out_stride_c;
    const float bias_val = has_bias == 1 ? b_buf[oc_tot] : 0.0;

    for (uint o0i = 0; o0i < o0n; ++o0i) {
        const uint out_base_o0 = out_base_n + o0i * out_stride_o0;
        for (uint o1i = 0; o1i < o1n; ++o1i) {
            const uint out_idx = out_base_o0 + o1i * out_stride_o1;
            y_buf[out_idx] = bias_val;
        }
    }

    for (uint ci = 0; ci < cinpg; ++ci) {
        const uint cin_tot = g_id * cinpg + ci;
        const uint w_base = cin_tot * w_stride_w0 + ocg_id * w_stride_w1;
        const uint in_base_c = n_id * in_stride_n + cin_tot * in_stride_c;

        for (uint i0 = 0; i0 < i0n; ++i0) {
            const uint in_base_i0 = in_base_c + i0 * in_stride_i0;
            for (uint i1 = 0; i1 < i1n; ++i1) {
                const uint in_idx = in_base_i0 + i1 * in_stride_i1;
                const float x_val = x_buf[in_idx];

                for (uint k0i = 0; k0i < k0; ++k0i) {
                    const uint w_k0 = w_base + k0i * w_stride_k0;
                    for (uint k1i = 0; k1i < k1; ++k1i) {
                        const uint w_idx = w_k0 + k1i * w_stride_k1;
                        const float w_val = w_buf[w_idx];

                        const int t0 = int(i0) * int(s0);
                        const int t1 = int(i1) * int(s1);
                        const int o0i = t0 - int(p0) + int(k0i) * int(d0);
                        const int o1i = t1 - int(p1) + int(k1i) * int(d1);

                        if (o0i >= 0 && uint(o0i) < o0n && o1i >= 0 && uint(o1i) < o1n) {
                            const uint out_idx = out_base_n + uint(o0i) * out_stride_o0 + uint(o1i) * out_stride_o1;
                            y_buf[out_idx] += x_val * w_val;
                        }
                    }
                }
            }
        }
    }
}
""")
        self.deconv3d_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; };
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; };
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; };

layout(constant_id = 0)  const float n_f  = 0.0;
layout(constant_id = 1)  const float cin_f  = 0.0;
layout(constant_id = 2)  const float cout_f = 0.0;
layout(constant_id = 3)  const float i0_f = 1.0;
layout(constant_id = 4)  const float i1_f = 1.0;
layout(constant_id = 5)  const float i2_f = 1.0;
layout(constant_id = 6)  const float o0_f = 1.0;
layout(constant_id = 7)  const float o1_f = 1.0;
layout(constant_id = 8)  const float o2_f = 1.0;
layout(constant_id = 9)  const float k0_f = 1.0;
layout(constant_id = 10) const float k1_f = 1.0;
layout(constant_id = 11) const float k2_f = 1.0;
layout(constant_id = 12) const float s0_f = 1.0;
layout(constant_id = 13) const float s1_f = 1.0;
layout(constant_id = 14) const float s2_f = 1.0;
layout(constant_id = 15) const float p0_f = 0.0;
layout(constant_id = 16) const float p1_f = 0.0;
layout(constant_id = 17) const float p2_f = 0.0;
layout(constant_id = 18) const float d0_f = 1.0;
layout(constant_id = 19) const float d1_f = 1.0;
layout(constant_id = 20) const float d2_f = 1.0;
layout(constant_id = 21) const float in_stride_n_f  = 1.0;
layout(constant_id = 22) const float in_stride_c_f  = 1.0;
layout(constant_id = 23) const float in_stride_i0_f = 1.0;
layout(constant_id = 24) const float in_stride_i1_f = 1.0;
layout(constant_id = 25) const float in_stride_i2_f = 1.0;
layout(constant_id = 26) const float out_stride_n_f  = 1.0;
layout(constant_id = 27) const float out_stride_c_f  = 1.0;
layout(constant_id = 28) const float out_stride_o0_f = 1.0;
layout(constant_id = 29) const float out_stride_o1_f = 1.0;
layout(constant_id = 30) const float out_stride_o2_f = 1.0;
layout(constant_id = 31) const float has_bias_f = 0.0;
layout(constant_id = 32) const float group_f  = 1.0;
layout(constant_id = 33) const float cinpg_f  = 1.0;
layout(constant_id = 34) const float coutpg_f = 1.0;
layout(constant_id = 35) const float w_stride_w0_f = 1.0;
layout(constant_id = 36) const float w_stride_w1_f = 1.0;
layout(constant_id = 37) const float w_stride_k0_f = 1.0;
layout(constant_id = 38) const float w_stride_k1_f = 1.0;
layout(constant_id = 39) const float w_stride_k2_f = 1.0;

void main() {
    uint n_id   = gl_GlobalInvocationID.x;
    uint ocg_id = gl_GlobalInvocationID.y;
    uint g_id   = gl_GlobalInvocationID.z;

    const uint n        = uint(n_f);
    const uint i0n      = uint(i0_f);
    const uint i1n      = uint(i1_f);
    const uint i2n      = uint(i2_f);
    const uint o0n      = uint(o0_f);
    const uint o1n      = uint(o1_f);
    const uint o2n      = uint(o2_f);
    const uint k0       = uint(k0_f);
    const uint k1       = uint(k1_f);
    const uint k2       = uint(k2_f);
    const uint s0       = uint(s0_f);
    const uint s1       = uint(s1_f);
    const uint s2       = uint(s2_f);
    const uint p0       = uint(p0_f);
    const uint p1       = uint(p1_f);
    const uint p2       = uint(p2_f);
    const uint d0       = uint(d0_f);
    const uint d1       = uint(d1_f);
    const uint d2       = uint(d2_f);
    const uint in_stride_n  = uint(in_stride_n_f);
    const uint in_stride_c  = uint(in_stride_c_f);
    const uint in_stride_i0 = uint(in_stride_i0_f);
    const uint in_stride_i1 = uint(in_stride_i1_f);
    const uint in_stride_i2 = uint(in_stride_i2_f);
    const uint out_stride_n  = uint(out_stride_n_f);
    const uint out_stride_c  = uint(out_stride_c_f);
    const uint out_stride_o0 = uint(out_stride_o0_f);
    const uint out_stride_o1 = uint(out_stride_o1_f);
    const uint out_stride_o2 = uint(out_stride_o2_f);
    const uint has_bias = uint(has_bias_f);
    const uint group     = uint(group_f);
    const uint cinpg     = uint(cinpg_f);
    const uint coutpg    = uint(coutpg_f);
    const uint w_stride_w0 = uint(w_stride_w0_f);
    const uint w_stride_w1 = uint(w_stride_w1_f);
    const uint w_stride_k0 = uint(w_stride_k0_f);
    const uint w_stride_k1 = uint(w_stride_k1_f);
    const uint w_stride_k2 = uint(w_stride_k2_f);


    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint out_base_n = n_id * out_stride_n + oc_tot * out_stride_c;
    const float bias_val = has_bias == 1 ? b_buf[oc_tot] : 0.0;

    for (uint o0i = 0; o0i < o0n; ++o0i) {
        const uint out_base_o0 = out_base_n + o0i * out_stride_o0;
        for (uint o1i = 0; o1i < o1n; ++o1i) {
            const uint out_base_o1 = out_base_o0 + o1i * out_stride_o1;
            for (uint o2i = 0; o2i < o2n; ++o2i) {
                const uint out_idx = out_base_o1 + o2i * out_stride_o2;
                y_buf[out_idx] = bias_val;
            }
        }
    }

    for (uint ci = 0; ci < cinpg; ++ci) {
        const uint cin_tot = g_id * cinpg + ci;
        const uint w_base = cin_tot * w_stride_w0 + ocg_id * w_stride_w1;
        const uint in_base_c = n_id * in_stride_n + cin_tot * in_stride_c;

        for (uint i0 = 0; i0 < i0n; ++i0) {
            const uint in_base_i0 = in_base_c + i0 * in_stride_i0;
            for (uint i1 = 0; i1 < i1n; ++i1) {
                const uint in_base_i1 = in_base_i0 + i1 * in_stride_i1;
                for (uint i2 = 0; i2 < i2n; ++i2) {
                    const uint in_idx = in_base_i1 + i2 * in_stride_i2;
                    const float x_val = x_buf[in_idx];

                    for (uint k0i = 0; k0i < k0; ++k0i) {
                        const uint w_k0 = w_base + k0i * w_stride_k0;
                        for (uint k1i = 0; k1i < k1; ++k1i) {
                            const uint w_k1 = w_k0 + k1i * w_stride_k1;
                            for (uint k2i = 0; k2i < k2; ++k2i) {
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
                                    o2i >= 0 && uint(o2i) < o2n) {
                                    const uint out_idx = out_base_n + 
                                        uint(o0i) * out_stride_o0 + 
                                        uint(o1i) * out_stride_o1 + 
                                        uint(o2i) * out_stride_o2;
                                    y_buf[out_idx] += x_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
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
                else:
                    pads_1.append(total_padding[i] - int_half)
                    pads_2.append(int_half)
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
        workgroup = (n, coutpg, g)

        if spatial_dims == 1:
            spec_consts = [
                n, cin, cout,
                i0, o0, k0, s0, p0, d0,
                in_stride_n, in_stride_c, in_stride_i0,
                out_stride_n, out_stride_c, out_stride_o0,
                has_bias, g, cinpg, coutpg,
                w_stride_w0, w_stride_w1, w_stride_k0,
            ]
            shader = self.deconv1d_shader
        elif spatial_dims == 2:
            spec_consts = [
                n, cin, cout,
                i0, i1, o0, o1,
                k0, k1,
                s0, s1,
                p0, p1,
                d0, d1,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1,
                has_bias, g, cinpg, coutpg,
                w_stride_w0, w_stride_w1, w_stride_k0, w_stride_k1,
            ]
            shader = self.deconv2d_shader
        else:
            spec_consts = [
                n, cin, cout,
                i0, i1, i2, o0, o1, o2,
                k0, k1, k2,
                s0, s1, s2,
                p0, p1, p2,
                d0, d1, d2,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1, in_stride_i2,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1, out_stride_o2,
                has_bias, g, cinpg, coutpg,
                w_stride_w0, w_stride_w1, w_stride_k0, w_stride_k1, w_stride_k2,
            ]
            shader = self.deconv3d_shader

        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, tensor_w, tensor_b, tensor_out],
            shader,
            workgroup,
            spec_consts,
            []
        ))

        return [(tensor_out, shape_out)]
