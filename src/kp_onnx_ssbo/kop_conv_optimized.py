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

        # ------------------------------------------------------------------
        # 1D Conv shader
        # params layout (uint):
        #  [0]=n  [1]=cin  [2]=cout  [3]=group  [4]=cinpg  [5]=coutpg  [6]=has_bias
        #  [7]=i0  [8]=o0  [9]=k0  [10]=s0  [11]=p0  [12]=d0
        #  [13]=outw  [14]=outh
        #  [15]=in_stride_n  [16]=in_stride_c  [17]=in_stride_i0
        #  [18]=out_stride_n  [19]=out_stride_c  [20]=out_stride_o0
        #  [21]=inc_i0
        # ------------------------------------------------------------------
        self.conv1d_gemm_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint gx4 = gl_GlobalInvocationID.x * 4u;
    uint ocg  = gl_GlobalInvocationID.y;
    uint ng   = gl_GlobalInvocationID.z;

    int n        = int(params[0]);
    int group    = int(params[3]);
    int cinpg    = int(params[4]);
    int coutpg   = int(params[5]);
    int has_bias = int(params[6]);

    int i0  = int(params[7]);
    int o0  = int(params[8]);
    int k0  = int(params[9]);
    int s0  = int(params[10]);
    int p0  = int(params[11]);
    int d0  = int(params[12]);

    int outw    = int(params[13]);
    int outh    = int(params[14]);
    int outsize = outw * outh;

    int in_stride_n   = int(params[15]);
    int in_stride_c   = int(params[16]);
    int in_stride_i0  = int(params[17]);

    int out_stride_n   = int(params[18]);
    int out_stride_c   = int(params[19]);
    int out_stride_o0  = int(params[20]);

    int inc_i0 = int(params[21]);

    if (ocg >= uint(coutpg) || ng >= uint(n * group) || gx4 >= uint(outsize)) return;

    int g_id   = int(ng % uint(group));
    int n_id   = int(ng / uint(group));
    int oc_tot = g_id * coutpg + int(ocg);

    float sum0 = (has_bias == 1) ? b_buf[uint(oc_tot)] : 0.0;
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

    int in_base_n  = n_id * in_stride_n;
    int w_base_oc  = oc_tot * cinpg * k0;

    for (int ci = 0; ci < cinpg; ++ci) {{
        int w_k        = w_base_oc + ci * k0;
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
        for (int k = 0; k < k0; ++k) {{
            float wv = w_buf[uint(wk)];
            if (idx0 < outsize && cur0 >= 0 && cur0 < i0) sum0 += x_buf[uint(in_i0_0)] * wv;
            if (idx1 < outsize && cur1 >= 0 && cur1 < i0) sum1 += x_buf[uint(in_i0_1)] * wv;
            if (idx2 < outsize && cur2 >= 0 && cur2 < i0) sum2 += x_buf[uint(in_i0_2)] * wv;
            if (idx3 < outsize && cur3 >= 0 && cur3 < i0) sum3 += x_buf[uint(in_i0_3)] * wv;
            cur0 += d0; in_i0_0 += inc_i0;
            cur1 += d0; in_i0_1 += inc_i0;
            cur2 += d0; in_i0_2 += inc_i0;
            cur3 += d0; in_i0_3 += inc_i0;
            wk += 1;
        }}
    }}

    int out_base = n_id * out_stride_n + oc_tot * out_stride_c;
    if (idx0 < outsize) y_buf[uint(out_base + sx0 * out_stride_o0)] = sum0;
    if (idx1 < outsize) y_buf[uint(out_base + sx1 * out_stride_o0)] = sum1;
    if (idx2 < outsize) y_buf[uint(out_base + sx2 * out_stride_o0)] = sum2;
    if (idx3 < outsize) y_buf[uint(out_base + sx3 * out_stride_o0)] = sum3;
}}
""")

        # ------------------------------------------------------------------
        # 2D Conv shader
        # params layout (uint):
        #  [0]=n  [1]=cin  [2]=cout  [3]=group  [4]=cinpg  [5]=coutpg  [6]=has_bias
        #  [7]=i0  [8]=i1  [9]=o0  [10]=o1  [11]=k0  [12]=k1
        #  [13]=s0  [14]=s1  [15]=p0  [16]=p1  [17]=d0  [18]=d1
        #  [19]=outw  [20]=outh
        #  [21]=in_stride_n  [22]=in_stride_c  [23]=in_stride_i0  [24]=in_stride_i1
        #  [25]=out_stride_n  [26]=out_stride_c  [27]=out_stride_o0  [28]=out_stride_o1
        #  [29]=inc_i0  [30]=inc_i1
        # ------------------------------------------------------------------
        self.conv2d_gemm_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint gx4 = gl_GlobalInvocationID.x * 4u;
    uint ocg  = gl_GlobalInvocationID.y;
    uint ng   = gl_GlobalInvocationID.z;

    int n        = int(params[0]);
    int group    = int(params[3]);
    int cinpg    = int(params[4]);
    int coutpg   = int(params[5]);
    int has_bias = int(params[6]);

    int i0  = int(params[7]);
    int i1  = int(params[8]);
    int o0  = int(params[9]);
    int o1  = int(params[10]);
    int k0  = int(params[11]);
    int k1  = int(params[12]);
    int s0  = int(params[13]);
    int s1  = int(params[14]);
    int p0  = int(params[15]);
    int p1  = int(params[16]);
    int d0  = int(params[17]);
    int d1  = int(params[18]);

    int outw    = int(params[19]);
    int outh    = int(params[20]);
    int outsize = outw * outh;

    int in_stride_n   = int(params[21]);
    int in_stride_c   = int(params[22]);
    int in_stride_i0  = int(params[23]);
    int in_stride_i1  = int(params[24]);

    int out_stride_n   = int(params[25]);
    int out_stride_c   = int(params[26]);
    int out_stride_o0  = int(params[27]);
    int out_stride_o1  = int(params[28]);

    int inc_i0 = int(params[29]);
    int inc_i1 = int(params[30]);

    if (ocg >= uint(coutpg) || ng >= uint(n * group) || gx4 >= uint(outsize)) return;

    int g_id   = int(ng % uint(group));
    int n_id   = int(ng / uint(group));
    int oc_tot = g_id * coutpg + int(ocg);

    float sum0 = (has_bias == 1) ? b_buf[uint(oc_tot)] : 0.0;
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

    int o0_0 = sy0; int o1_0 = sx0;
    int o0_1 = sy1; int o1_1 = sx1;
    int o0_2 = sy2; int o1_2 = sx2;
    int o0_3 = sy3; int o1_3 = sx3;

    int i0s0 = o0_0 * s0 - p0;
    int i0s1 = o0_1 * s0 - p0;
    int i0s2 = o0_2 * s0 - p0;
    int i0s3 = o0_3 * s0 - p0;

    int i1s0 = o1_0 * s1 - p1;
    int i1s1 = o1_1 * s1 - p1;
    int i1s2 = o1_2 * s1 - p1;
    int i1s3 = o1_3 * s1 - p1;

    int in_base_n  = n_id * in_stride_n;
    int w_base_oc  = oc_tot * cinpg * (k0 * k1);

    for (int ci = 0; ci < cinpg; ++ci) {{
        int w_k0       = w_base_oc + ci * (k0 * k1);
        int in_base_nc = in_base_n + (g_id * cinpg + ci) * in_stride_c;

        int in_i0_0 = in_base_nc + i0s0 * in_stride_i0;
        int in_i0_1 = in_base_nc + i0s1 * in_stride_i0;
        int in_i0_2 = in_base_nc + i0s2 * in_stride_i0;
        int in_i0_3 = in_base_nc + i0s3 * in_stride_i0;

        int cur0_0 = i0s0;
        int cur0_1 = i0s1;
        int cur0_2 = i0s2;
        int cur0_3 = i0s3;

        for (int k0i = 0; k0i < k0; ++k0i) {{
            int in_i1_0 = in_i0_0 + i1s0 * in_stride_i1;
            int in_i1_1 = in_i0_1 + i1s1 * in_stride_i1;
            int in_i1_2 = in_i0_2 + i1s2 * in_stride_i1;
            int in_i1_3 = in_i0_3 + i1s3 * in_stride_i1;

            int cur1_0 = i1s0;
            int cur1_1 = i1s1;
            int cur1_2 = i1s2;
            int cur1_3 = i1s3;

            int w_k1 = w_k0;
            for (int k1i = 0; k1i < k1; ++k1i) {{
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
            }}

            cur0_0 += d0; in_i0_0 += inc_i0; w_k0 += k1;
            cur0_1 += d0; in_i0_1 += inc_i0;
            cur0_2 += d0; in_i0_2 += inc_i0;
            cur0_3 += d0; in_i0_3 += inc_i0;
        }}
    }}

    int out_base = n_id * out_stride_n + oc_tot * out_stride_c;
    if (idx0 < outsize) y_buf[uint(out_base + o0_0 * out_stride_o0 + o1_0 * out_stride_o1)] = sum0;
    if (idx1 < outsize) y_buf[uint(out_base + o0_1 * out_stride_o0 + o1_1 * out_stride_o1)] = sum1;
    if (idx2 < outsize) y_buf[uint(out_base + o0_2 * out_stride_o0 + o1_2 * out_stride_o1)] = sum2;
    if (idx3 < outsize) y_buf[uint(out_base + o0_3 * out_stride_o0 + o1_3 * out_stride_o1)] = sum3;
}}
""")

        # ------------------------------------------------------------------
        # 3D Conv shader
        # params layout (uint):
        #  [0]=n  [1]=cin  [2]=cout  [3]=group  [4]=cinpg  [5]=coutpg  [6]=has_bias
        #  [7]=i0  [8]=i1  [9]=i2
        #  [10]=o0  [11]=o1  [12]=o2
        #  [13]=k0  [14]=k1  [15]=k2
        #  [16]=s0  [17]=s1  [18]=s2
        #  [19]=p0  [20]=p1  [21]=p2
        #  [22]=d0  [23]=d1  [24]=d2
        #  [25]=outw  [26]=outh
        #  [27]=in_stride_n  [28]=in_stride_c
        #  [29]=in_stride_i0  [30]=in_stride_i1  [31]=in_stride_i2
        #  [32]=out_stride_n  [33]=out_stride_c
        #  [34]=out_stride_o0  [35]=out_stride_o1  [36]=out_stride_o2
        #  [37]=k1k2  [38]=k0k1k2
        #  [39]=inc_i0  [40]=inc_i1  [41]=inc_i2
        # ------------------------------------------------------------------
        self.conv3d_gemm_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout(std430, set = 0, binding = 0) readonly buffer XBuf {{ float x_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer WBuf {{ float w_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer BBuf {{ float b_buf[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OBuf {{ float y_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint gx4 = gl_GlobalInvocationID.x * 4u;
    uint ocg  = gl_GlobalInvocationID.y;
    uint ng   = gl_GlobalInvocationID.z;

    int n        = int(params[0]);
    int group    = int(params[3]);
    int cinpg    = int(params[4]);
    int coutpg   = int(params[5]);
    int has_bias = int(params[6]);

    int i0 = int(params[7]);
    int i1 = int(params[8]);
    int i2 = int(params[9]);
    int o0 = int(params[10]);
    int o1 = int(params[11]);
    int o2 = int(params[12]);
    int k0 = int(params[13]);
    int k1 = int(params[14]);
    int k2 = int(params[15]);
    int s0 = int(params[16]);
    int s1 = int(params[17]);
    int s2 = int(params[18]);
    int p0 = int(params[19]);
    int p1 = int(params[20]);
    int p2 = int(params[21]);
    int d0 = int(params[22]);
    int d1 = int(params[23]);
    int d2 = int(params[24]);

    int outw    = int(params[25]);
    int outh    = int(params[26]);
    int outsize = outw * outh;

    int in_stride_n   = int(params[27]);
    int in_stride_c   = int(params[28]);
    int in_stride_i0  = int(params[29]);
    int in_stride_i1  = int(params[30]);
    int in_stride_i2  = int(params[31]);

    int out_stride_n   = int(params[32]);
    int out_stride_c   = int(params[33]);
    int out_stride_o0  = int(params[34]);
    int out_stride_o1  = int(params[35]);
    int out_stride_o2  = int(params[36]);

    int k1k2   = int(params[37]);
    int k0k1k2 = int(params[38]);
    int inc_i0  = int(params[39]);
    int inc_i1  = int(params[40]);
    int inc_i2  = int(params[41]);

    if (ocg >= uint(coutpg) || ng >= uint(n * group) || gx4 >= uint(outsize)) return;

    int g_id   = int(ng % uint(group));
    int n_id   = int(ng / uint(group));
    int oc_tot = g_id * coutpg + int(ocg);

    float sum0 = (has_bias == 1) ? b_buf[uint(oc_tot)] : 0.0;
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

    int o2_0 = sx0; int o2_1 = sx1; int o2_2 = sx2; int o2_3 = sx3;

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

    int in_base_n  = n_id * in_stride_n;
    int w_base_oc  = oc_tot * cinpg * k0k1k2;

    for (int ci = 0; ci < cinpg; ++ci) {{
        int w_base_ci  = w_base_oc + ci * k0k1k2;
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
        for (int k0i = 0; k0i < k0; ++k0i) {{
            int in_i1_0 = in_i0_0 + i1s0 * in_stride_i1;
            int in_i1_1 = in_i0_1 + i1s1 * in_stride_i1;
            int in_i1_2 = in_i0_2 + i1s2 * in_stride_i1;
            int in_i1_3 = in_i0_3 + i1s3 * in_stride_i1;

            int cur1_0 = i1s0;
            int cur1_1 = i1s1;
            int cur1_2 = i1s2;
            int cur1_3 = i1s3;

            int w_k1 = w_k0;
            for (int k1i = 0; k1i < k1; ++k1i) {{
                int in_i2_0 = in_i1_0 + i2s0 * in_stride_i2;
                int in_i2_1 = in_i1_1 + i2s1 * in_stride_i2;
                int in_i2_2 = in_i1_2 + i2s2 * in_stride_i2;
                int in_i2_3 = in_i1_3 + i2s3 * in_stride_i2;

                int cur2_0 = i2s0;
                int cur2_1 = i2s1;
                int cur2_2 = i2s2;
                int cur2_3 = i2s3;

                int w_k2 = w_k1;
                for (int k2i = 0; k2i < k2; ++k2i) {{
                    float wv = w_buf[uint(w_k2)];
                    if (idx0 < outsize && cur0_0 >= 0 && cur0_0 < i0 && cur1_0 >= 0 && cur1_0 < i1 && cur2_0 >= 0 && cur2_0 < i2) sum0 += x_buf[uint(in_i2_0)] * wv;
                    if (idx1 < outsize && cur0_1 >= 0 && cur0_1 < i0 && cur1_1 >= 0 && cur1_1 < i1 && cur2_1 >= 0 && cur2_1 < i2) sum1 += x_buf[uint(in_i2_1)] * wv;
                    if (idx2 < outsize && cur0_2 >= 0 && cur0_2 < i0 && cur1_2 >= 0 && cur1_2 < i1 && cur2_2 >= 0 && cur2_2 < i2) sum2 += x_buf[uint(in_i2_2)] * wv;
                    if (idx3 < outsize && cur0_3 >= 0 && cur0_3 < i0 && cur1_3 >= 0 && cur1_3 < i1 && cur2_3 >= 0 && cur2_3 < i2) sum3 += x_buf[uint(in_i2_3)] * wv;
                    cur2_0 += d2; in_i2_0 += inc_i2; w_k2 += 1;
                    cur2_1 += d2; in_i2_1 += inc_i2;
                    cur2_2 += d2; in_i2_2 += inc_i2;
                    cur2_3 += d2; in_i2_3 += inc_i2;
                }}

                cur1_0 += d1; in_i1_0 += inc_i1; w_k1 += k2;
                cur1_1 += d1; in_i1_1 += inc_i1;
                cur1_2 += d1; in_i1_2 += inc_i1;
                cur1_3 += d1; in_i1_3 += inc_i1;
            }}

            cur0_0 += d0; in_i0_0 += inc_i0; w_k0 += k1k2;
            cur0_1 += d0; in_i0_1 += inc_i0;
            cur0_2 += d0; in_i0_2 += inc_i0;
            cur0_3 += d0; in_i0_3 += inc_i0;
        }}
    }}

    int out_base = n_id * out_stride_n + oc_tot * out_stride_c;
    if (idx0 < outsize) y_buf[uint(out_base + o0_0 * out_stride_o0 + o1_0 * out_stride_o1 + o2_0 * out_stride_o2)] = sum0;
    if (idx1 < outsize) y_buf[uint(out_base + o0_1 * out_stride_o0 + o1_1 * out_stride_o1 + o2_1 * out_stride_o2)] = sum1;
    if (idx2 < outsize) y_buf[uint(out_base + o0_2 * out_stride_o0 + o1_2 * out_stride_o1 + o2_2 * out_stride_o2)] = sum2;
    if (idx3 < outsize) y_buf[uint(out_base + o0_3 * out_stride_o0 + o1_3 * out_stride_o1 + o2_3 * out_stride_o2)] = sum3;
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

        # auto_pad 处理：使用空间维度 in_spatial[i] 计算 pad
        if auto_pad == "VALID":
            pads_final = [0] * (spatial_dims * 2)
            out_spatial = [
                int((in_spatial[i] - kernel[i]) // strides[i] + 1)
                for i in range(spatial_dims)
            ]
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            head, tail = [], []
            for i in range(spatial_dims):
                d = in_spatial[i]
                target_size = (d + strides[i] - 1) // strides[i]
                pad_needed = max(0, (target_size - 1) * strides[i] + kernel[i] - d)
                if auto_pad == "SAME_LOWER":
                    pad_head = (pad_needed + 1) // 2
                else:
                    pad_head = pad_needed // 2
                pad_tail = pad_needed - pad_head
                head.append(int(pad_head))
                tail.append(int(pad_tail))
            pads_final = head + tail
            out_spatial = [
                int((in_spatial[i] + pads_final[i] + pads_final[i + spatial_dims] - kernel[i]) // strides[i] + 1)
                for i in range(spatial_dims)
            ]
        else:
            out_spatial, pads_final = self._get_output_shape_explicit_padding(
                in_spatial, kernel, strides, pads_in, dilations)

        # 步长计算（flatten，从内到外）
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
            workgroup = (
                (grid_x + LOCAL_X_3D - 1) // LOCAL_X_3D,
                (coutpg + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                (n * g + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
            )
            inc_i0 = d0 * in_stride_i0
            params = np.array([
                n, cin, cout, g, cinpg, coutpg, has_bias,   # [0..6]
                i0, o0, k0, s0, p0, d0,                     # [7..12]
                outw, outh,                                  # [13..14]
                in_stride_n, in_stride_c, in_stride_i0,     # [15..17]
                out_stride_n, out_stride_c, out_stride_o0,  # [18..20]
                inc_i0,                                      # [21]
            ], dtype=np.uint32)
            shader = self.conv1d_gemm_shader

        elif spatial_dims == 2:
            outw = o1
            outh = o0
            grid_x = (outw * outh + 3) // 4
            workgroup = (
                (grid_x + LOCAL_X_3D - 1) // LOCAL_X_3D,
                (coutpg + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                (n * g + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
            )
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            params = np.array([
                n, cin, cout, g, cinpg, coutpg, has_bias,             # [0..6]
                i0, i1, o0, o1, k0, k1, s0, s1, p0, p1, d0, d1,      # [7..18]
                outw, outh,                                            # [19..20]
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1, # [21..24]
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1, # [25..28]
                inc_i0, inc_i1,                                        # [29..30]
            ], dtype=np.uint32)
            shader = self.conv2d_gemm_shader

        else:  # spatial_dims == 3
            outw = o2
            outh = o0 * o1
            grid_x = (outw * outh + 3) // 4
            workgroup = (
                (grid_x + LOCAL_X_3D - 1) // LOCAL_X_3D,
                (coutpg + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                (n * g + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
            )
            k1k2   = k1 * k2
            k0k1k2 = k0 * k1k2
            inc_i0 = d0 * in_stride_i0
            inc_i1 = d1 * in_stride_i1
            inc_i2 = d2  # in_stride_i2 == 1
            params = np.array([
                n, cin, cout, g, cinpg, coutpg, has_bias,                      # [0..6]
                i0, i1, i2, o0, o1, o2, k0, k1, k2, s0, s1, s2, p0, p1, p2,   # [7..21]
                d0, d1, d2,                                                     # [22..24]
                outw, outh,                                                     # [25..26]
                in_stride_n, in_stride_c, in_stride_i0, in_stride_i1, in_stride_i2,  # [27..31]
                out_stride_n, out_stride_c, out_stride_o0, out_stride_o1, out_stride_o2,  # [32..36]
                k1k2, k0k1k2, inc_i0, inc_i1, inc_i2,                          # [37..41]
            ], dtype=np.uint32)
            shader = self.conv3d_gemm_shader

        # 按规范：param_in 定义后立即 sync 到 GPU
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, tensor_w, tensor_b, tensor_out, param_in],
            shader,
            workgroup,
        ))

        return [(tensor_out, shape_out)]

