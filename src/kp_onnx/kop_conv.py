import kp
import numpy as np
from .shader_utils import compile_source


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
        self.conv1d_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; }; // 输入特征图X
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; }; // 卷积核W
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; }; // 偏置B（可选）
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; }; // 输出特征图Y

layout(constant_id = 0)  const float n_f  = 0.0;  // batch size（样本数量，N）
layout(constant_id = 1)  const float cin_f  = 0.0;  // 输入通道数（Cin）
layout(constant_id = 2)  const float cout_f = 0.0;  // 输出通道数（Cout）

layout(constant_id = 3)  const float i0_f = 1.0;  // 输入特征图的1D空间长度（如序列长度，I）
layout(constant_id = 4)  const float o0_f = 1.0;  // 输出特征图的1D空间长度（O）
layout(constant_id = 5)  const float k0_f = 1.0;  // 卷积核的1D尺寸（K）
layout(constant_id = 6)  const float s0_f = 1.0;  // 1D卷积的步长（stride）
layout(constant_id = 7)  const float p0_f = 0.0;  // 1D卷积的填充（padding）
layout(constant_id = 8)  const float d0_f = 1.0;  // 1D卷积的膨胀率（dilation）

layout(constant_id = 9)  const float in_stride_n_f  = 1.0;  // 样本维度的步长（N维度）
layout(constant_id = 10) const float in_stride_c_f  = 1.0;  // 输入通道维度的步长（Cin维度）
layout(constant_id = 11) const float in_stride_i0_f = 1.0;  // 1D空间维度的步长（I维度）

layout(constant_id = 12) const float out_stride_n_f  = 1.0;  // 样本维度的步长（N维度）
layout(constant_id = 13) const float out_stride_c_f  = 1.0;  // 输出通道维度的步长（Cout维度）
layout(constant_id = 14) const float out_stride_o0_f = 1.0;  // 1D空间维度的步长（O维度）

layout(constant_id = 15) const float inc_i0_f  = 1.0;  // 1D空间维度的索引增量
layout(constant_id = 16) const float has_bias_f = 0.0;  // 是否使用偏置（0：无，1：有）

layout(constant_id = 17) const float group_f  = 1.0;  // 组卷积的组数
layout(constant_id = 18) const float cinpg_f  = 1.0;  // 每组的输入通道数
layout(constant_id = 19) const float coutpg_f = 1.0;  // 每组的输出通道数

void main() {
    uint n_id  = gl_GlobalInvocationID.x;   // batch索引
    uint ocg_id = gl_GlobalInvocationID.y;  // 组内输出通道索引
    uint g_id  = gl_GlobalInvocationID.z;   // 组索引

    const uint n        = uint(n_f);
    const uint i0       = uint(i0_f);
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
    const uint inc_i0    = uint(inc_i0_f);
    const uint has_bias  = uint(has_bias_f);
    const uint group     = uint(group_f);
    const uint cinpg     = uint(cinpg_f);
    const uint coutpg    = uint(coutpg_f);

    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint w_base_oc = oc_tot * cinpg * k0;
    const uint in_base_n = n_id * in_stride_n;
    const uint out_base = n_id * out_stride_n + oc_tot * out_stride_c;

    for (uint o0i = 0; o0i < o0n; ++o0i) {
        const uint out_idx = out_base + o0i * out_stride_o0;

        const int i0s = int(o0i * s0) - int(p0);

        float acc = has_bias == 1 ? b_buf[oc_tot] : 0.0;

        for (uint ci = 0; ci < cinpg; ++ci) {
            const uint in_base_ci = in_base_n + (g_id * cinpg + ci) * in_stride_c;
            uint in_i = in_base_ci + uint(i0s) * in_stride_i0;

            const uint w_base_ci = w_base_oc + ci * k0;
            uint w_k = w_base_ci;

            int cur_i = i0s;
            for (uint kk = 0; kk < k0; ++kk) {
                if (cur_i >= 0 && cur_i < int(i0)) {
                    acc += x_buf[in_i] * w_buf[w_k];
                }
                cur_i += int(d0);
                in_i  += inc_i0;
                w_k   += 1;
            }
        }

        y_buf[out_idx] = acc;
    }
}
""")
        self.conv2d_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; }; // 输入特征图X
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; }; // 卷积核W
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; }; // 偏置B（可选）
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; }; // 输出特征图Y

layout(constant_id = 0)  const float n_f  = 0.0;  // batch size（样本数量，N）
layout(constant_id = 1)  const float cin_f  = 0.0;  // 输入通道数（Cin）
layout(constant_id = 2)  const float cout_f = 0.0;  // 输出通道数（Cout）

layout(constant_id = 3)  const float i0_f = 1.0;  // 输入特征图的高度（H）
layout(constant_id = 4)  const float i1_f = 1.0;  // 输入特征图的宽度（W）
layout(constant_id = 5)  const float o0_f = 1.0;  // 输出特征图的高度（Oh）
layout(constant_id = 6)  const float o1_f = 1.0;  // 输出特征图的宽度（Ow）

layout(constant_id = 7)  const float k0_f = 1.0;  // 卷积核的高度（Kh）
layout(constant_id = 8)  const float k1_f = 1.0;  // 卷积核的宽度（Kw）

layout(constant_id = 9)  const float s0_f = 1.0;  // 高度方向的步长（stride_h）
layout(constant_id = 10) const float s1_f = 1.0;  // 宽度方向的步长（stride_w）

layout(constant_id = 11) const float p0_f = 0.0;  // 高度方向的填充（padding_h）
layout(constant_id = 12) const float p1_f = 0.0;  // 宽度方向的填充（padding_w）

layout(constant_id = 13) const float d0_f = 1.0;  // 高度方向的膨胀率（dilation_h）
layout(constant_id = 14) const float d1_f = 1.0;  // 宽度方向的膨胀率（dilation_w）

layout(constant_id = 15) const float in_stride_n_f  = 1.0;  // 样本维度步长（N）
layout(constant_id = 16) const float in_stride_c_f  = 1.0;  // 输入通道维度步长（Cin）
layout(constant_id = 17) const float in_stride_i0_f = 1.0;  // 高度维度步长（H）
layout(constant_id = 18) const float in_stride_i1_f = 1.0;  // 宽度维度步长（W）

layout(constant_id = 19) const float out_stride_n_f  = 1.0;  // 样本维度步长（N）
layout(constant_id = 20) const float out_stride_c_f  = 1.0;  // 输出通道维度步长（Cout）
layout(constant_id = 21) const float out_stride_o0_f = 1.0;  // 高度维度步长（Oh）
layout(constant_id = 22) const float out_stride_o1_f = 1.0;  // 宽度维度步长（Ow）

layout(constant_id = 23) const float inc_i0_f  = 1.0;  // 高度方向索引增量（d0 * in_stride_i0）
layout(constant_id = 24) const float inc_i1_f  = 1.0;  // 宽度方向索引增量（d1 * in_stride_i1）
layout(constant_id = 25) const float has_bias_f = 0.0;  // 是否使用偏置（0：无，1：有）

layout(constant_id = 26) const float group_f  = 1.0;  // 组卷积的组数（group）
layout(constant_id = 27) const float cinpg_f  = 1.0;  // 每组的输入通道数（Cin / group）
layout(constant_id = 28) const float coutpg_f = 1.0;  // 每组的输出通道数（Cout / group）

void main() {
    uint n_id  = gl_GlobalInvocationID.x;   // batch索引
    uint ocg_id = gl_GlobalInvocationID.y;  // 组内输出通道索引
    uint g_id  = gl_GlobalInvocationID.z;   // 组索引

    const uint n        = uint(n_f);
    const uint i0       = uint(i0_f);
    const uint i1       = uint(i1_f);
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
    const uint inc_i0    = uint(inc_i0_f);
    const uint inc_i1    = uint(inc_i1_f);
    const uint has_bias  = uint(has_bias_f);
    const uint group     = uint(group_f);
    const uint cinpg     = uint(cinpg_f);
    const uint coutpg    = uint(coutpg_f);
    
    const uint k0k1      = k0 * k1;
    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint w_base_oc = oc_tot * cinpg * k0k1;
    const uint in_base_n = n_id * in_stride_n;
    const uint in_base_g = in_base_n + g_id * cinpg * in_stride_c;
    const uint out_base = n_id * out_stride_n + oc_tot * out_stride_c;

    for (uint o0i = 0; o0i < o0n; ++o0i) {
        const uint out_base_o0 = out_base + o0i * out_stride_o0;

        const int i0s = int(o0i * s0) - int(p0);

        for (uint o1i = 0; o1i < o1n; ++o1i) {
            const uint out_idx = out_base_o0 + o1i * out_stride_o1;

            const int i1s = int(o1i * s1) - int(p1);

            float acc = has_bias == 1 ? b_buf[oc_tot] : 0.0;

            for (uint ci = 0; ci < cinpg; ++ci) {
                const uint in_base_ci = in_base_g + ci * in_stride_c;
                uint in_i0 = in_base_ci + uint(i0s) * in_stride_i0;

                const uint w_base_ci = w_base_oc + ci * k0k1;
                uint w_k0 = w_base_ci;

                int cur_i0 = i0s;
                for (uint k0i = 0; k0i < k0; ++k0i) {
                    if (cur_i0 >= 0 && cur_i0 < int(i0)) {
                        uint in_i1 = in_i0 + uint(i1s) * in_stride_i1;
                        uint w_k1 = w_k0;

                        int cur_i1 = i1s;
                        for (uint k1i = 0; k1i < k1; ++k1i) {
                            if (cur_i1 >= 0 && cur_i1 < int(i1)) {
                                acc += x_buf[in_i1] * w_buf[w_k1];
                            }
                            cur_i1 += int(d1);
                            in_i1  += inc_i1;
                            w_k1   += 1;
                        }
                    }
                    cur_i0 += int(d0);
                    in_i0  += inc_i0;
                    w_k0   += k1;
                }
            }

            y_buf[out_idx] = acc;
        }
    }
}
""")
        self.conv3d_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; }; // 输入特征图X
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; }; // 卷积核W
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; }; // 偏置B（可选）
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; }; // 输出特征图Y

layout(constant_id = 0)  const float n_f  = 0.0;  // batch size（样本数量，N）
layout(constant_id = 1)  const float cin_f  = 0.0;  // 输入通道数（Cin）
layout(constant_id = 2)  const float cout_f = 0.0;  // 输出通道数（Cout）

layout(constant_id = 3)  const float i0_f = 1.0;  // 输入特征图的深度（D）
layout(constant_id = 4)  const float i1_f = 1.0;  // 输入特征图的高度（H）
layout(constant_id = 5)  const float i2_f = 1.0;  // 输入特征图的宽度（W）
layout(constant_id = 6)  const float o0_f = 1.0;  // 输出特征图的深度（Od）
layout(constant_id = 7)  const float o1_f = 1.0;  // 输出特征图的高度（Oh）
layout(constant_id = 8)  const float o2_f = 1.0;  // 输出特征图的宽度（Ow）

layout(constant_id = 9)  const float k0_f = 1.0;  // 卷积核的深度（Kd）
layout(constant_id = 10) const float k1_f = 1.0;  // 卷积核的高度（Kh）
layout(constant_id = 11) const float k2_f = 1.0;  // 卷积核的宽度（Kw）

layout(constant_id = 12) const float s0_f = 1.0;  // 深度方向的步长（stride_d）
layout(constant_id = 13) const float s1_f = 1.0;  // 高度方向的步长（stride_h）
layout(constant_id = 14) const float s2_f = 1.0;  // 宽度方向的步长（stride_w）

layout(constant_id = 15) const float p0_f = 0.0;  // 深度方向的填充（padding_d）
layout(constant_id = 16) const float p1_f = 0.0;  // 高度方向的填充（padding_h）
layout(constant_id = 17) const float p2_f = 0.0;  // 宽度方向的填充（padding_w）

layout(constant_id = 18) const float d0_f = 1.0;  // 深度方向的膨胀率（dilation_d）
layout(constant_id = 19) const float d1_f = 1.0;  // 高度方向的膨胀率（dilation_h）
layout(constant_id = 20) const float d2_f = 1.0;  // 宽度方向的膨胀率（dilation_w）

layout(constant_id = 21) const float in_stride_n_f  = 1.0;  // 样本维度步长（N）
layout(constant_id = 22) const float in_stride_c_f  = 1.0;  // 输入通道维度步长（Cin）
layout(constant_id = 23) const float in_stride_i0_f = 1.0;  // 深度维度步长（D）
layout(constant_id = 24) const float in_stride_i1_f = 1.0;  // 高度维度步长（H）
layout(constant_id = 25) const float in_stride_i2_f = 1.0;  // 宽度维度步长（W）

layout(constant_id = 26) const float out_stride_n_f  = 1.0;  // 样本维度步长（N）
layout(constant_id = 27) const float out_stride_c_f  = 1.0;  // 输出通道维度步长（Cout）
layout(constant_id = 28) const float out_stride_o0_f = 1.0;  // 深度维度步长（Od）
layout(constant_id = 29) const float out_stride_o1_f = 1.0;  // 高度维度步长（Oh）
layout(constant_id = 30) const float out_stride_o2_f = 1.0;  // 宽度维度步长（Ow）

layout(constant_id = 31) const float k1k2_f    = 1.0;  // 卷积核高度×宽度（Kh×Kw）
layout(constant_id = 32) const float k0k1k2_f  = 1.0;  // 卷积核深度×高度×宽度（Kd×Kh×Kw）
layout(constant_id = 33) const float inc_i0_f  = 1.0;  // 深度方向索引增量（d0 * in_stride_i0）
layout(constant_id = 34) const float inc_i1_f  = 1.0;  // 高度方向索引增量（d1 * in_stride_i1）
layout(constant_id = 35) const float inc_i2_f  = 1.0;  // 宽度方向索引增量（d2 * in_stride_i2）

layout(constant_id = 36) const float has_bias_f = 0.0;  // 是否使用偏置（0：无，1：有）

layout(constant_id = 37) const float group_f  = 1.0;  // 组卷积的组数（group）
layout(constant_id = 38) const float cinpg_f  = 1.0;  // 每组的输入通道数（Cin / group）
layout(constant_id = 39) const float coutpg_f = 1.0;  // 每组的输出通道数（Cout / group）

void main() {
    uint n_id  = gl_GlobalInvocationID.x;   // batch索引
    uint ocg_id = gl_GlobalInvocationID.y;  // 组内输出通道索引
    uint g_id  = gl_GlobalInvocationID.z;   // 组索引

    const uint n        = uint(n_f);
    const uint i0       = uint(i0_f);
    const uint i1       = uint(i1_f);
    const uint i2       = uint(i2_f);
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
    const uint k1k2     = uint(k1k2_f);
    const uint k0k1k2   = uint(k0k1k2_f);
    const uint inc_i0   = uint(inc_i0_f);
    const uint inc_i1   = uint(inc_i1_f);
    const uint inc_i2   = uint(inc_i2_f);
    const uint has_bias = uint(has_bias_f);
    const uint group    = uint(group_f);
    const uint cinpg    = uint(cinpg_f);
    const uint coutpg   = uint(coutpg_f);


    const uint oc_tot = g_id * coutpg + ocg_id;
    const uint w_base_oc = oc_tot * cinpg * k0k1k2;
    const uint in_base_n = n_id * in_stride_n;
    const uint in_base_g = in_base_n + g_id * cinpg * in_stride_c;
    const uint out_base = n_id * out_stride_n + oc_tot * out_stride_c;

    for (uint o0i = 0; o0i < o0n; ++o0i) {
        const uint out_base_o0 = out_base + o0i * out_stride_o0;

        const int i0s = int(o0i * s0) - int(p0);

        for (uint o1i = 0; o1i < o1n; ++o1i) {
            const uint out_base_o1 = out_base_o0 + o1i * out_stride_o1;

            const int i1s = int(o1i * s1) - int(p1);

            for (uint o2i = 0; o2i < o2n; ++o2i) {
                const uint out_idx = out_base_o1 + o2i * out_stride_o2;

                const int i2s = int(o2i * s2) - int(p2);

                float acc = has_bias == 1 ? b_buf[oc_tot] : 0.0;

                for (uint ci = 0; ci < cinpg; ++ci) {
                    const uint in_base_ci = in_base_g + ci * in_stride_c;
                    uint in_i0 = in_base_ci + uint(i0s) * in_stride_i0;
                    const uint w_base_ci = w_base_oc + ci * k0k1k2;
                    uint w_k0 = w_base_ci;

                    int cur_i0 = i0s;
                    for (uint k0i = 0; k0i < k0; ++k0i) {
                        if (cur_i0 >= 0 && cur_i0 < int(i0)) {
                            uint in_i1 = in_i0 + uint(i1s) * in_stride_i1;
                            uint w_k1 = w_k0;

                            int cur_i1 = i1s;
                            for (uint k1i = 0; k1i < k1; ++k1i) {
                                if (cur_i1 >= 0 && cur_i1 < int(i1)) {
                                    uint in_i2 = in_i1 + uint(i2s) * in_stride_i2;
                                    uint w_k2 = w_k1;

                                    int cur_i2 = i2s; 
                                    for (uint k2i = 0; k2i < k2; ++k2i) {
                                        if (cur_i2 >= 0 && cur_i2 < int(i2)) {
                                            acc += x_buf[in_i2] * w_buf[w_k2];
                                        }

                                        cur_i2 += int(d2);
                                        in_i2  += inc_i2;
                                        w_k2   += 1;
                                    }
                                }
                                cur_i1 += int(d1);
                                in_i1  += inc_i1;
                                w_k1   += k2;
                            }
                        }
                        cur_i0 += int(d0);
                        in_i0  += inc_i0;
                        w_k0   += k1k2;
                    }
                }

                y_buf[out_idx] = acc;
            }
        }
    }
}
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

        # 根据 ONNX 规范，必要时在末端增加额外 pad 以对齐计算窗口
        pads_new = pad_list[:]
        for dim in range(dims):
            eff_k = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
            actual = (out_shape[dim] - 1) * strides_spatial[dim] + eff_k
            extra = actual - input_spatial_shape[dim] - pad_list[dim] - pad_list[dims + dim]
            if extra > 0:
                pads_new[dim] += extra // 2
                pads_new[dims + dim] += extra - extra // 2
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
            tensor_b = self.manager.tensor(np.zeros(1, dtype=np.float32))
            updated_tensors.append(tensor_b)

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
        workgroup = (n, coutpg, g)

        if spatial_dims == 1:
            # 1D shader 常量
            inc_i0 = d0 * in_stride_i0
            spec_consts = [
                n, cin, cout,
                i0, o0, k0, s0, p0, d0,
                in_stride_n, in_stride_c, in_stride_i0,
                out_stride_n, out_stride_c, out_stride_o0,
                inc_i0, has_bias,
                g, cinpg, coutpg,
            ]
            shader = self.conv1d_shader
        elif spatial_dims == 2:
            # 2D shader 常量
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            spec_consts = [
                n, cin, cout,
                i0, i1, o0, o1,
                k0, k1,
                s0, s1,
                p0, p1,
                d0, d1,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1,
                inc_i0, inc_i1, has_bias,
                g, cinpg, coutpg,
            ]
            shader = self.conv2d_shader
        else:
            # 3D shader 常量
            k1k2 = k1 * k2
            k0k1k2 = k0 * k1k2
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            inc_i2 = d2
            spec_consts = [
                n, cin, cout,
                i0, i1, i2, o0, o1, o2,
                k0, k1, k2,
                s0, s1, s2,
                p0, p1, p2,
                d0, d1, d2,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1, in_stride_i2,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1, out_stride_o2,
                k1k2, k0k1k2, inc_i0, inc_i1, inc_i2,
                has_bias,
                g, cinpg, coutpg,
            ]
            shader = self.conv3d_shader

        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, tensor_w, tensor_b, tensor_out],
            shader,
            workgroup,
            spec_consts,
            []
        ))

        return [(tensor_out, shape_out)]