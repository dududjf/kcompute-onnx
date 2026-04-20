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
        self.conv1d_gemm_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; };
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; };
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; };

layout(constant_id = 0)  const float n_f  = 1.0;
layout(constant_id = 1)  const float cin_f  = 1.0;
layout(constant_id = 2)  const float cout_f = 1.0;
layout(constant_id = 3)  const float group_f  = 1.0;
layout(constant_id = 4)  const float cinpg_f  = 1.0;
layout(constant_id = 5)  const float coutpg_f = 1.0;
layout(constant_id = 6)  const float has_bias_f = 0.0;

layout(constant_id = 7)  const float i0_f = 1.0;
layout(constant_id = 8)  const float o0_f = 1.0;
layout(constant_id = 9)  const float k0_f = 1.0;
layout(constant_id = 10) const float s0_f = 1.0;
layout(constant_id = 11) const float p0_f = 0.0;
layout(constant_id = 12) const float d0_f = 1.0;

layout(constant_id = 13) const float outw_f = 1.0; // = o0
layout(constant_id = 14) const float outh_f = 1.0; // = 1

layout(constant_id = 15) const float in_stride_n_f  = 1.0;
layout(constant_id = 16) const float in_stride_c_f  = 1.0;
layout(constant_id = 17) const float in_stride_i0_f = 1.0;

layout(constant_id = 18) const float out_stride_n_f  = 1.0;
layout(constant_id = 19) const float out_stride_c_f  = 1.0;
layout(constant_id = 20) const float out_stride_o0_f = 1.0;

layout(constant_id = 21) const float inc_i0_f  = 1.0; // d0 * in_stride_i0

void main(){
    uint gx4 = gl_GlobalInvocationID.x * 4u;
    uint ocg = gl_GlobalInvocationID.y;
    uint ng  = gl_GlobalInvocationID.z;

    int n      = int(n_f);
    int group  = int(group_f);
    int cinpg  = int(cinpg_f);
    int coutpg = int(coutpg_f);
    int has_bias = int(has_bias_f);

    int i0  = int(i0_f);
    int o0  = int(o0_f);
    int k0  = int(k0_f);
    int s0  = int(s0_f);
    int p0  = int(p0_f);
    int d0  = int(d0_f);

    int outw = int(outw_f);
    int outh = int(outh_f);
    int outsize = outw * outh; // = o0

    int in_stride_n  = int(in_stride_n_f);
    int in_stride_c  = int(in_stride_c_f);
    int in_stride_i0 = int(in_stride_i0_f);

    int out_stride_n  = int(out_stride_n_f);
    int out_stride_c  = int(out_stride_c_f);
    int out_stride_o0 = int(out_stride_o0_f);

    int inc_i0 = int(inc_i0_f);

    if (ocg >= uint(coutpg) || ng >= uint(n * group) || gx4 >= uint(outsize)) return;

    int g_id = int(ng % uint(group));
    int n_id = int(ng / uint(group));
    int oc_tot = g_id * coutpg + int(ocg);

    float sum0 = has_bias == 1 ? b_buf[uint(oc_tot)] : 0.0;
    float sum1 = sum0;
    float sum2 = sum0;
    float sum3 = sum0;

    int idx0 = int(gx4) + 0;
    int idx1 = int(gx4) + 1;
    int idx2 = int(gx4) + 2;
    int idx3 = int(gx4) + 3;

    int sx0 = (idx0 < outsize) ? (idx0 % outw) : 0;
    int sx1 = (idx1 < outsize) ? (idx1 % outw) : 0;
    int sx2 = (idx2 < outsize) ? (idx2 % outw) : 0;
    int sx3 = (idx3 < outsize) ? (idx3 % outw) : 0;

    int i0s0 = sx0 * s0 - p0;
    int i0s1 = sx1 * s0 - p0;
    int i0s2 = sx2 * s0 - p0;
    int i0s3 = sx3 * s0 - p0;

    int in_base_n = n_id * in_stride_n;

    int w_base_oc = oc_tot * cinpg * k0;

    for (int ci = 0; ci < cinpg; ++ci) {
        int w_k = w_base_oc + ci * k0;
        int in_base_nc = in_base_n + (g_id * cinpg + ci) * in_stride_c;

        int in_i0_0 = in_base_nc + i0s0 * in_stride_i0;
        int in_i0_1 = in_base_nc + i0s1 * in_stride_i0;
        int in_i0_2 = in_base_nc + i0s2 * in_stride_i0;
        int in_i0_3 = in_base_nc + i0s3 * in_stride_i0;

        int cur0 = i0s0;
        int cur1 = i0s1;
        int cur2 = i0s2;
        int cur3 = i0s3;

        int wk = w_k;
        for (int k = 0; k < k0; ++k) {
            float wv = w_buf[uint(wk)];
            if (idx0 < outsize && cur0 >= 0 && cur0 < i0) sum0 += x_buf[uint(in_i0_0)] * wv;
            if (idx1 < outsize && cur1 >= 0 && cur1 < i0) sum1 += x_buf[uint(in_i0_1)] * wv;
            if (idx2 < outsize && cur2 >= 0 && cur2 < i0) sum2 += x_buf[uint(in_i0_2)] * wv;
            if (idx3 < outsize && cur3 >= 0 && cur3 < i0) sum3 += x_buf[uint(in_i0_3)] * wv;
            cur0 += d0; 
            in_i0_0 += inc_i0;
            cur1 += d0; 
            in_i0_1 += inc_i0;
            cur2 += d0; 
            in_i0_2 += inc_i0;
            cur3 += d0; 
            in_i0_3 += inc_i0;
            wk += 1;
        }
    }

    int out_base = n_id * out_stride_n + oc_tot * out_stride_c;
    if (idx0 < outsize) { int o0i = sx0; y_buf[uint(out_base + o0i * out_stride_o0)] = sum0; }
    if (idx1 < outsize) { int o0i = sx1; y_buf[uint(out_base + o0i * out_stride_o0)] = sum1; }
    if (idx2 < outsize) { int o0i = sx2; y_buf[uint(out_base + o0i * out_stride_o0)] = sum2; }
    if (idx3 < outsize) { int o0i = sx3; y_buf[uint(out_base + o0i * out_stride_o0)] = sum3; }
}
""")
        self.conv2d_gemm_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; };
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; };
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; };

layout(constant_id = 0)  const float n_f  = 1.0;
layout(constant_id = 1)  const float cin_f  = 1.0;
layout(constant_id = 2)  const float cout_f = 1.0;
layout(constant_id = 3)  const float group_f  = 1.0;
layout(constant_id = 4)  const float cinpg_f  = 1.0;
layout(constant_id = 5)  const float coutpg_f = 1.0;
layout(constant_id = 6)  const float has_bias_f = 0.0;

layout(constant_id = 7)  const float i0_f = 1.0;
layout(constant_id = 8)  const float i1_f = 1.0;
layout(constant_id = 9)  const float o0_f = 1.0;
layout(constant_id = 10) const float o1_f = 1.0;
layout(constant_id = 11) const float k0_f = 1.0;
layout(constant_id = 12) const float k1_f = 1.0;
layout(constant_id = 13) const float s0_f = 1.0;
layout(constant_id = 14) const float s1_f = 1.0;
layout(constant_id = 15) const float p0_f = 0.0;
layout(constant_id = 16) const float p1_f = 0.0;
layout(constant_id = 17) const float d0_f = 1.0;
layout(constant_id = 18) const float d1_f = 1.0;

layout(constant_id = 19) const float outw_f = 1.0; // = o1
layout(constant_id = 20) const float outh_f = 1.0; // = o0

layout(constant_id = 21) const float in_stride_n_f  = 1.0;
layout(constant_id = 22) const float in_stride_c_f  = 1.0;
layout(constant_id = 23) const float in_stride_i0_f = 1.0;
layout(constant_id = 24) const float in_stride_i1_f = 1.0;

layout(constant_id = 25) const float out_stride_n_f  = 1.0;
layout(constant_id = 26) const float out_stride_c_f  = 1.0;
layout(constant_id = 27) const float out_stride_o0_f = 1.0;
layout(constant_id = 28) const float out_stride_o1_f = 1.0;

layout(constant_id = 29) const float inc_i0_f  = 1.0;
layout(constant_id = 30) const float inc_i1_f  = 1.0;

void main(){
    uint gx4 = gl_GlobalInvocationID.x * 4u;
    uint ocg = gl_GlobalInvocationID.y;
    uint ng  = gl_GlobalInvocationID.z;

    int n      = int(n_f);
    int group  = int(group_f);
    int cinpg  = int(cinpg_f);
    int coutpg = int(coutpg_f);
    int has_bias = int(has_bias_f);

    int i0 = int(i0_f);
    int i1 = int(i1_f);
    int o0 = int(o0_f);
    int o1 = int(o1_f);
    int k0 = int(k0_f);
    int k1 = int(k1_f);
    int s0 = int(s0_f);
    int s1 = int(s1_f);
    int p0 = int(p0_f);
    int p1 = int(p1_f);
    int d0 = int(d0_f);
    int d1 = int(d1_f);

    int outw = int(outw_f);
    int outh = int(outh_f);
    int outsize = outw * outh; // = o0 * o1

    int in_stride_n  = int(in_stride_n_f);
    int in_stride_c  = int(in_stride_c_f);
    int in_stride_i0 = int(in_stride_i0_f);
    int in_stride_i1 = int(in_stride_i1_f);

    int out_stride_n  = int(out_stride_n_f);
    int out_stride_c  = int(out_stride_c_f);
    int out_stride_o0 = int(out_stride_o0_f);
    int out_stride_o1 = int(out_stride_o1_f);

    int inc_i0 = int(inc_i0_f);
    int inc_i1 = int(inc_i1_f);

    if (ocg >= uint(coutpg) || ng >= uint(n * group) || gx4 >= uint(outsize)) return;

    int g_id = int(ng % uint(group));
    int n_id = int(ng / uint(group));
    int oc_tot = g_id * coutpg + int(ocg);

    float sum0 = has_bias == 1 ? b_buf[uint(oc_tot)] : 0.0;
    float sum1 = sum0;
    float sum2 = sum0;
    float sum3 = sum0;

    int idx0 = int(gx4) + 0;
    int idx1 = int(gx4) + 1;
    int idx2 = int(gx4) + 2;
    int idx3 = int(gx4) + 3;

    int sy0 = (idx0 < outsize) ? (idx0 / outw) : 0;
    int sy1 = (idx1 < outsize) ? (idx1 / outw) : 0;
    int sy2 = (idx2 < outsize) ? (idx2 / outw) : 0;
    int sy3 = (idx3 < outsize) ? (idx3 / outw) : 0;

    int sx0 = (idx0 < outsize) ? (idx0 % outw) : 0;
    int sx1 = (idx1 < outsize) ? (idx1 % outw) : 0;
    int sx2 = (idx2 < outsize) ? (idx2 % outw) : 0;
    int sx3 = (idx3 < outsize) ? (idx3 % outw) : 0;

    int o0_0 = sy0, o1_0 = sx0;
    int o0_1 = sy1, o1_1 = sx1;
    int o0_2 = sy2, o1_2 = sx2;
    int o0_3 = sy3, o1_3 = sx3;

    int i0s0 = o0_0 * s0 - p0;
    int i0s1 = o0_1 * s0 - p0;
    int i0s2 = o0_2 * s0 - p0;
    int i0s3 = o0_3 * s0 - p0;

    int i1s0 = o1_0 * s1 - p1;
    int i1s1 = o1_1 * s1 - p1;
    int i1s2 = o1_2 * s1 - p1;
    int i1s3 = o1_3 * s1 - p1;

    int in_base_n = n_id * in_stride_n;
    int w_base_oc = oc_tot * cinpg * (k0 * k1);

    for (int ci = 0; ci < cinpg; ++ci) {
        int w_k0 = w_base_oc + ci * (k0 * k1);
        int in_base_nc = in_base_n + (g_id * cinpg + ci) * in_stride_c;

        int in_i0_0 = in_base_nc + i0s0 * in_stride_i0;
        int in_i0_1 = in_base_nc + i0s1 * in_stride_i0;
        int in_i0_2 = in_base_nc + i0s2 * in_stride_i0;
        int in_i0_3 = in_base_nc + i0s3 * in_stride_i0;

        int cur0_0 = i0s0;
        int cur0_1 = i0s1;
        int cur0_2 = i0s2;
        int cur0_3 = i0s3;

        for (int k0i = 0; k0i < k0; ++k0i) {
            int in_i1_0 = in_i0_0 + i1s0 * in_stride_i1;
            int in_i1_1 = in_i0_1 + i1s1 * in_stride_i1;
            int in_i1_2 = in_i0_2 + i1s2 * in_stride_i1;
            int in_i1_3 = in_i0_3 + i1s3 * in_stride_i1;

            int cur1_0 = i1s0;
            int cur1_1 = i1s1;
            int cur1_2 = i1s2;
            int cur1_3 = i1s3;

            int w_k1 = w_k0;
            for (int k1i = 0; k1i < k1; ++k1i) {
                float wv = w_buf[uint(w_k1)];
                if (idx0 < outsize && cur0_0 >= 0 && cur0_0 < i0 && cur1_0 >= 0 && cur1_0 < i1) sum0 += x_buf[uint(in_i1_0)] * wv;
                if (idx1 < outsize && cur0_1 >= 0 && cur0_1 < i0 && cur1_1 >= 0 && cur1_1 < i1) sum1 += x_buf[uint(in_i1_1)] * wv;
                if (idx2 < outsize && cur0_2 >= 0 && cur0_2 < i0 && cur1_2 >= 0 && cur1_2 < i1) sum2 += x_buf[uint(in_i1_2)] * wv;
                if (idx3 < outsize && cur0_3 >= 0 && cur0_3 < i0 && cur1_3 >= 0 && cur1_3 < i1) sum3 += x_buf[uint(in_i1_3)] * wv;
                cur1_0 += d1; in_i1_0 += inc_i1;
                cur1_1 += d1; in_i1_1 += inc_i1;
                cur1_2 += d1; in_i1_2 += inc_i1;
                cur1_3 += d1; in_i1_3 += inc_i1;
                w_k1 += 1;
            }

            cur0_0 += d0; in_i0_0 += inc_i0; w_k0 += k1;
            cur0_1 += d0; in_i0_1 += inc_i0;
            cur0_2 += d0; in_i0_2 += inc_i0;
            cur0_3 += d0; in_i0_3 += inc_i0;
        }
    }

    int out_base = n_id * out_stride_n + oc_tot * out_stride_c;
    if (idx0 < outsize) y_buf[uint(out_base + o0_0 * out_stride_o0 + o1_0 * out_stride_o1)] = sum0;
    if (idx1 < outsize) y_buf[uint(out_base + o0_1 * out_stride_o0 + o1_1 * out_stride_o1)] = sum1;
    if (idx2 < outsize) y_buf[uint(out_base + o0_2 * out_stride_o0 + o1_2 * out_stride_o1)] = sum2;
    if (idx3 < outsize) y_buf[uint(out_base + o0_3 * out_stride_o0 + o1_3 * out_stride_o1)] = sum3;
}
""")
        self.conv3d_gemm_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set = 0, binding = 0) readonly buffer XBuf { float x_buf[]; };
layout(set = 0, binding = 1) readonly buffer WBuf { float w_buf[]; };
layout(set = 0, binding = 2) readonly buffer BBuf { float b_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OBuf { float y_buf[]; };

layout(constant_id = 0)  const float n_f  = 1.0;
layout(constant_id = 1)  const float cin_f  = 1.0;
layout(constant_id = 2)  const float cout_f = 1.0;
layout(constant_id = 3)  const float group_f  = 1.0;
layout(constant_id = 4)  const float cinpg_f  = 1.0;
layout(constant_id = 5)  const float coutpg_f = 1.0;
layout(constant_id = 6)  const float has_bias_f = 0.0;

layout(constant_id = 7)  const float i0_f = 1.0;
layout(constant_id = 8)  const float i1_f = 1.0;
layout(constant_id = 9)  const float i2_f = 1.0;
layout(constant_id = 10) const float o0_f = 1.0;
layout(constant_id = 11) const float o1_f = 1.0;
layout(constant_id = 12) const float o2_f = 1.0;
layout(constant_id = 13) const float k0_f = 1.0;
layout(constant_id = 14) const float k1_f = 1.0;
layout(constant_id = 15) const float k2_f = 1.0;
layout(constant_id = 16) const float s0_f = 1.0;
layout(constant_id = 17) const float s1_f = 1.0;
layout(constant_id = 18) const float s2_f = 1.0;
layout(constant_id = 19) const float p0_f = 0.0;
layout(constant_id = 20) const float p1_f = 0.0;
layout(constant_id = 21) const float p2_f = 0.0;
layout(constant_id = 22) const float d0_f = 1.0;
layout(constant_id = 23) const float d1_f = 1.0;
layout(constant_id = 24) const float d2_f = 1.0;

layout(constant_id = 25) const float outw_f = 1.0; // = o2
layout(constant_id = 26) const float outh_f = 1.0; // = o0 * o1

layout(constant_id = 27) const float in_stride_n_f  = 1.0;
layout(constant_id = 28) const float in_stride_c_f  = 1.0;
layout(constant_id = 29) const float in_stride_i0_f = 1.0;
layout(constant_id = 30) const float in_stride_i1_f = 1.0;
layout(constant_id = 31) const float in_stride_i2_f = 1.0;

layout(constant_id = 32) const float out_stride_n_f  = 1.0;
layout(constant_id = 33) const float out_stride_c_f  = 1.0;
layout(constant_id = 34) const float out_stride_o0_f = 1.0;
layout(constant_id = 35) const float out_stride_o1_f = 1.0;
layout(constant_id = 36) const float out_stride_o2_f = 1.0;

layout(constant_id = 37) const float k1k2_f   = 1.0;
layout(constant_id = 38) const float k0k1k2_f = 1.0;
layout(constant_id = 39) const float inc_i0_f = 1.0;
layout(constant_id = 40) const float inc_i1_f = 1.0;
layout(constant_id = 41) const float inc_i2_f = 1.0;

void main(){
    uint gx4 = gl_GlobalInvocationID.x * 4u;
    uint ocg = gl_GlobalInvocationID.y;
    uint ng  = gl_GlobalInvocationID.z;

    int n      = int(n_f);
    int group  = int(group_f);
    int cinpg  = int(cinpg_f);
    int coutpg = int(coutpg_f);
    int has_bias = int(has_bias_f);

    int i0 = int(i0_f);
    int i1 = int(i1_f);
    int i2 = int(i2_f);
    int o0 = int(o0_f);
    int o1 = int(o1_f);
    int o2 = int(o2_f);
    int k0 = int(k0_f);
    int k1 = int(k1_f);
    int k2 = int(k2_f);
    int s0 = int(s0_f);
    int s1 = int(s1_f);
    int s2 = int(s2_f);
    int p0 = int(p0_f);
    int p1 = int(p1_f);
    int p2 = int(p2_f);
    int d0 = int(d0_f);
    int d1 = int(d1_f);
    int d2 = int(d2_f);

    int outw = int(outw_f);
    int outh = int(outh_f);
    int outsize = outw * outh; // = o0*o1*o2

    int in_stride_n  = int(in_stride_n_f);
    int in_stride_c  = int(in_stride_c_f);
    int in_stride_i0 = int(in_stride_i0_f);
    int in_stride_i1 = int(in_stride_i1_f);
    int in_stride_i2 = int(in_stride_i2_f);

    int out_stride_n  = int(out_stride_n_f);
    int out_stride_c  = int(out_stride_c_f);
    int out_stride_o0 = int(out_stride_o0_f);
    int out_stride_o1 = int(out_stride_o1_f);
    int out_stride_o2 = int(out_stride_o2_f);

    int k1k2   = int(k1k2_f);
    int k0k1k2 = int(k0k1k2_f);
    int inc_i0 = int(inc_i0_f);
    int inc_i1 = int(inc_i1_f);
    int inc_i2 = int(inc_i2_f);

    if (ocg >= uint(coutpg) || ng >= uint(n * group) || gx4 >= uint(outsize)) return;

    int g_id = int(ng % uint(group));
    int n_id = int(ng / uint(group));
    int oc_tot = g_id * coutpg + int(ocg);

    float sum0 = has_bias == 1 ? b_buf[uint(oc_tot)] : 0.0;
    float sum1 = sum0;
    float sum2 = sum0;
    float sum3 = sum0;

    int idx0 = int(gx4) + 0;
    int idx1 = int(gx4) + 1;
    int idx2 = int(gx4) + 2;
    int idx3 = int(gx4) + 3;

    int sy0 = (idx0 < outsize) ? (idx0 / outw) : 0;
    int sy1 = (idx1 < outsize) ? (idx1 / outw) : 0;
    int sy2 = (idx2 < outsize) ? (idx2 / outw) : 0;
    int sy3 = (idx3 < outsize) ? (idx3 / outw) : 0;

    int sx0 = (idx0 < outsize) ? (idx0 % outw) : 0;
    int sx1 = (idx1 < outsize) ? (idx1 % outw) : 0;
    int sx2 = (idx2 < outsize) ? (idx2 % outw) : 0;
    int sx3 = (idx3 < outsize) ? (idx3 % outw) : 0;

    // 解包 sy -> (o0i, o1i)
    int o0_0 = (sy0 < (o0 * o1)) ? (sy0 / o1) : 0;
    int o0_1 = (sy1 < (o0 * o1)) ? (sy1 / o1) : 0;
    int o0_2 = (sy2 < (o0 * o1)) ? (sy2 / o1) : 0;
    int o0_3 = (sy3 < (o0 * o1)) ? (sy3 / o1) : 0;

    int o1_0 = (sy0 < (o0 * o1)) ? (sy0 % o1) : 0;
    int o1_1 = (sy1 < (o0 * o1)) ? (sy1 % o1) : 0;
    int o1_2 = (sy2 < (o0 * o1)) ? (sy2 % o1) : 0;
    int o1_3 = (sy3 < (o0 * o1)) ? (sy3 % o1) : 0;

    int o2_0 = sx0, o2_1 = sx1, o2_2 = sx2, o2_3 = sx3;

    int i0s0 = o0_0 * s0 - p0;
    int i0s1 = o0_1 * s0 - p0;
    int i0s2 = o0_2 * s0 - p0;
    int i0s3 = o0_3 * s0 - p0;

    int i1s0 = o1_0 * s1 - p1;
    int i1s1 = o1_1 * s1 - p1;
    int i1s2 = o1_2 * s1 - p1;
    int i1s3 = o1_3 * s1 - p1;

    int i2s0 = o2_0 * s2 - p2;
    int i2s1 = o2_1 * s2 - p2;
    int i2s2 = o2_2 * s2 - p2;
    int i2s3 = o2_3 * s2 - p2;

    int in_base_n = n_id * in_stride_n;
    int w_base_oc = oc_tot * cinpg * k0k1k2;

    for (int ci = 0; ci < cinpg; ++ci) {
        int w_base_ci = w_base_oc + ci * k0k1k2;
        int in_base_nc = in_base_n + (g_id * cinpg + ci) * in_stride_c;

        int in_i0_0 = in_base_nc + i0s0 * in_stride_i0;
        int in_i0_1 = in_base_nc + i0s1 * in_stride_i0;
        int in_i0_2 = in_base_nc + i0s2 * in_stride_i0;
        int in_i0_3 = in_base_nc + i0s3 * in_stride_i0;

        int cur0_0 = i0s0;
        int cur0_1 = i0s1;
        int cur0_2 = i0s2;
        int cur0_3 = i0s3;

        int w_k0 = w_base_ci;
        for (int k0i = 0; k0i < k0; ++k0i) {
            int in_i1_0 = in_i0_0 + i1s0 * in_stride_i1;
            int in_i1_1 = in_i0_1 + i1s1 * in_stride_i1;
            int in_i1_2 = in_i0_2 + i1s2 * in_stride_i1;
            int in_i1_3 = in_i0_3 + i1s3 * in_stride_i1;

            int cur1_0 = i1s0;
            int cur1_1 = i1s1;
            int cur1_2 = i1s2;
            int cur1_3 = i1s3;

            int w_k1 = w_k0;
            for (int k1i = 0; k1i < k1; ++k1i) {
                int in_i2_0 = in_i1_0 + i2s0 * in_stride_i2;
                int in_i2_1 = in_i1_1 + i2s1 * in_stride_i2;
                int in_i2_2 = in_i1_2 + i2s2 * in_stride_i2;
                int in_i2_3 = in_i1_3 + i2s3 * in_stride_i2;

                int cur2_0 = i2s0;
                int cur2_1 = i2s1;
                int cur2_2 = i2s2;
                int cur2_3 = i2s3;

                int w_k2 = w_k1;
                for (int k2i = 0; k2i < k2; ++k2i) {
                    float wv = w_buf[uint(w_k2)];
                    if (idx0 < outsize && cur0_0 >= 0 && cur0_0 < i0 && cur1_0 >= 0 && cur1_0 < i1 && cur2_0 >= 0 && cur2_0 < i2) sum0 += x_buf[uint(in_i2_0)] * wv;
                    if (idx1 < outsize && cur0_1 >= 0 && cur0_1 < i0 && cur1_1 >= 0 && cur1_1 < i1 && cur2_1 >= 0 && cur2_1 < i2) sum1 += x_buf[uint(in_i2_1)] * wv;
                    if (idx2 < outsize && cur0_2 >= 0 && cur0_2 < i0 && cur1_2 >= 0 && cur1_2 < i1 && cur2_2 >= 0 && cur2_2 < i2) sum2 += x_buf[uint(in_i2_2)] * wv;
                    if (idx3 < outsize && cur0_3 >= 0 && cur0_3 < i0 && cur1_3 >= 0 && cur1_3 < i1 && cur2_3 >= 0 && cur2_3 < i2) sum3 += x_buf[uint(in_i2_3)] * wv;
                    cur2_0 += d2; in_i2_0 += inc_i2; w_k2 += 1;
                    cur2_1 += d2; in_i2_1 += inc_i2;
                    cur2_2 += d2; in_i2_2 += inc_i2;
                    cur2_3 += d2; in_i2_3 += inc_i2;
                }

                cur1_0 += d1; in_i1_0 += inc_i1; w_k1 += k2;
                cur1_1 += d1; in_i1_1 += inc_i1;
                cur1_2 += d1; in_i1_2 += inc_i1;
                cur1_3 += d1; in_i1_3 += inc_i1;
            }

            cur0_0 += d0; in_i0_0 += inc_i0; w_k0 += k1k2;
            cur0_1 += d0; in_i0_1 += inc_i0;
            cur0_2 += d0; in_i0_2 += inc_i0;
            cur0_3 += d0; in_i0_3 += inc_i0;
        }
    }

    int out_base = n_id * out_stride_n + oc_tot * out_stride_c;
    if (idx0 < outsize) y_buf[uint(out_base + o0_0 * out_stride_o0 + o1_0 * out_stride_o1 + o2_0 * out_stride_o2)] = sum0;
    if (idx1 < outsize) y_buf[uint(out_base + o0_1 * out_stride_o0 + o1_1 * out_stride_o1 + o2_1 * out_stride_o2)] = sum1;
    if (idx2 < outsize) y_buf[uint(out_base + o0_2 * out_stride_o0 + o1_2 * out_stride_o1 + o2_2 * out_stride_o2)] = sum2;
    if (idx3 < outsize) y_buf[uint(out_base + o0_3 * out_stride_o0 + o1_3 * out_stride_o1 + o2_3 * out_stride_o2)] = sum3;
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
        return out_shape, pad_list

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

        # 对齐 op_conv_optimized.py 的 auto_pad 行为（包括 VALID）
        if auto_pad in ("SAME_UPPER", "SAME_LOWER", "VALID"):
            head = []
            tail = []
            for i in range(spatial_dims):
                d = shape_x[i]  # 注意：与参考实现一致地使用 X.shape[i]
                target_size = (d + strides[i] - 1) // strides[i]
                pad_needed = (target_size - 1) * strides[i] + kernel[i] - d
                if auto_pad == "SAME_LOWER":
                    pad_head = (pad_needed + 1) // 2
                else:
                    pad_head = pad_needed // 2
                pad_tail = pad_needed - pad_head
                head.append(int(pad_head))
                tail.append(int(pad_tail))
            pads_final = head + tail
            # 依据 im2col 公式计算输出空间尺寸
            out_spatial = [
                int((in_spatial[i] + pads_final[i] + pads_final[i + spatial_dims] - kernel[i]) // strides[i] + 1)
                for i in range(spatial_dims)
            ]
        else:
            out_spatial, pads_final = self._get_output_shape_explicit_padding(in_spatial, kernel, strides, pads_in, dilations)

        # 步长（flatten）
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

        shape_out = [n, cout] + out_spatial
        tensor_out = self.manager.tensor(np.zeros(int(np.prod(shape_out)), dtype=np.float32))
        updated_tensors.append(tensor_out)

        if spatial_dims == 1:
            outw = o0
            outh = 1
            grid_x = (outw * outh + 3) // 4
            workgroup = (grid_x, coutpg, n * g)
            inc_i0 = d0 * in_stride_i0
            spec_consts = [
                n, cin, cout, g, cinpg, coutpg, has_bias,
                i0, o0, k0, s0, p0, d0,
                outw, outh,
                in_stride_n, in_stride_c, in_stride_i0,
                out_stride_n, out_stride_c, out_stride_o0,
                inc_i0,
            ]
            shader = self.conv1d_gemm_shader
        elif spatial_dims == 2:
            outw = o1
            outh = o0
            grid_x = (outw * outh + 3) // 4
            workgroup = (grid_x, coutpg, n * g)
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            spec_consts = [
                n, cin, cout, g, cinpg, coutpg, has_bias,
                i0, i1, o0, o1, k0, k1, s0, s1, p0, p1, d0, d1,
                outw, outh,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1,
                inc_i0, inc_i1,
            ]
            shader = self.conv2d_gemm_shader
        else:
            outw = o2
            outh = o0 * o1
            grid_x = (outw * outh + 3) // 4
            workgroup = (grid_x, coutpg, n * g)
            k1k2 = k1 * k2
            k0k1k2 = k0 * k1k2
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            inc_i2 = d2
            spec_consts = [
                n, cin, cout, g, cinpg, coutpg, has_bias,
                i0, i1, i2, o0, o1, o2, k0, k1, k2, s0, s1, s2, p0, p1, p2, d0, d1, d2,
                outw, outh,
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1, in_stride_i2,
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1, out_stride_o2,
                k1k2, k0k1k2, inc_i0, inc_i1, inc_i2,
            ]
            shader = self.conv3d_gemm_shader

        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, tensor_w, tensor_b, tensor_out],
            shader,
            workgroup,
            spec_consts,
            []
        ))

        return [(tensor_out, shape_out)]
