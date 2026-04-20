import kp
import numpy as np
from .shader_utils import compile_source


class GruOp:
    """
    GRU (Gated Recurrent Unit) operator.
    
    Supports:
    - num_directions = 1 (forward/reverse) or 2 (bidirectional)
    - linear_before_reset = 0 or 1
    - layout = 0 (seq_length, batch, input) or 1 (batch, seq_length, input)
    - direction = 'forward', 'reverse', or 'bidirectional'
    
    Shader organization (6 shaders):
    - Unidirectional + linear_before_reset=0: shader_uni_phase1, shader_uni_phase2
    - Bidirectional + linear_before_reset=0: shader_bi_phase1, shader_bi_phase2
    - Unidirectional + linear_before_reset=1: shader_uni_linear
    - Bidirectional + linear_before_reset=1: shader_bi_linear
    """
    
    def __init__(self, manager: kp.Manager, 
                 activation_alpha=None,
                 activation_beta=None,
                 activations=None,
                 clip=None,
                 direction='forward',
                 hidden_size=None,
                 layout=0,
                 linear_before_reset=0
                 ):
        self.manager = manager
        self.hidden_size = hidden_size
        self.linear_before_reset = linear_before_reset
        self.layout = layout
        self.direction = direction
        self.number_of_gates = 3
        
        # ============================================================
        # Shader 1: Unidirectional Phase1 (linear_before_reset=0)
        # Workgroup: (batch_size, hidden_size, 1)
        # Computes z, r gates and stores r*H for candidate state
        # ============================================================
        self.shader_uni_phase1 = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_x { float X[]; };
layout (binding = 1) readonly buffer buf_w { float W[]; };
layout (binding = 2) readonly buffer buf_r { float R[]; };
layout (binding = 3) readonly buffer buf_b { float B[]; };
layout (binding = 4) readonly buffer buf_h { float H[]; };
layout (binding = 5) buffer buf_rh { float RH[]; };
layout (binding = 6) buffer buf_z { float Z[]; };

layout (constant_id = 0) const float input_size_f = 0;
layout (constant_id = 1) const float hidden_size_f = 0;
layout (constant_id = 2) const float x_offset_f = 0;
layout (constant_id = 3) const float x_stride_b_f = 0;
layout (constant_id = 4) const float w_dir_offset_f = 0;
layout (constant_id = 5) const float r_dir_offset_f = 0;
layout (constant_id = 6) const float b_dir_offset_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    
    uint input_size = uint(input_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint x_offset = uint(x_offset_f);
    uint x_stride_b = uint(x_stride_b_f);
    uint w_dir_offset = uint(w_dir_offset_f);
    uint r_dir_offset = uint(r_dir_offset_f);
    uint b_dir_offset = uint(b_dir_offset_f);
    
    uint x_base = x_offset + b * x_stride_b;
    uint h_idx = b * hidden_size + j;
    uint h_base = b * hidden_size;
    
    uint w_z_off = w_dir_offset + j * input_size;
    uint w_r_off = w_dir_offset + (hidden_size + j) * input_size;
    uint r_z_off = r_dir_offset + j * hidden_size;
    uint r_r_off = r_dir_offset + (hidden_size + j) * hidden_size;
    
    float b_z = B[b_dir_offset + j] + B[b_dir_offset + 3 * hidden_size + j];
    float b_r = B[b_dir_offset + hidden_size + j] + B[b_dir_offset + 4 * hidden_size + j];
    
    // z gate: z_t = σ(W_z x_t + R_z h_{t-1} + b_z)
    float z_acc = b_z;
    for (uint xi = x_base, wi = w_z_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
        z_acc += X[xi] * W[wi];
    }
    for (uint hi = h_base, ri = r_z_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
        z_acc += H[hi] * R[ri];
    }
    float z = 1.0 / (1.0 + exp(-z_acc));
    Z[h_idx] = z;
    
    // r gate: r_t = σ(W_r x_t + R_r h_{t-1} + b_r)
    float r_acc = b_r;
    for (uint xi = x_base, wi = w_r_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
        r_acc += X[xi] * W[wi];
    }
    for (uint hi = h_base, ri = r_r_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
        r_acc += H[hi] * R[ri];
    }
    float r = 1.0 / (1.0 + exp(-r_acc));
    
    // RH = r_t ⊙ h_{t-1}
    RH[h_idx] = r * H[h_idx];
}
""")

        # ============================================================
        # Shader 2: Unidirectional Phase2 (linear_before_reset=0)
        # Workgroup: (batch_size, hidden_size, 1)
        # Computes h_cand and updates H, Y
        # ============================================================
        self.shader_uni_phase2 = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_x { float X[]; };
layout (binding = 1) readonly buffer buf_w { float W[]; };
layout (binding = 2) readonly buffer buf_r { float R[]; };
layout (binding = 3) readonly buffer buf_b { float B[]; };
layout (binding = 4) buffer buf_h { float H[]; };
layout (binding = 5) readonly buffer buf_rh { float RH[]; };
layout (binding = 6) readonly buffer buf_z { float Z[]; };
layout (binding = 7) buffer buf_y { float Y[]; };

layout (constant_id = 0) const float input_size_f = 0;
layout (constant_id = 1) const float hidden_size_f = 0;
layout (constant_id = 2) const float x_offset_f = 0;
layout (constant_id = 3) const float x_stride_b_f = 0;
layout (constant_id = 4) const float y_offset_f = 0;
layout (constant_id = 5) const float y_stride_b_f = 0;
layout (constant_id = 6) const float w_dir_offset_f = 0;
layout (constant_id = 7) const float r_dir_offset_f = 0;
layout (constant_id = 8) const float b_dir_offset_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    
    uint input_size = uint(input_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint x_offset = uint(x_offset_f);
    uint x_stride_b = uint(x_stride_b_f);
    uint y_offset = uint(y_offset_f);
    uint y_stride_b = uint(y_stride_b_f);
    uint w_dir_offset = uint(w_dir_offset_f);
    uint r_dir_offset = uint(r_dir_offset_f);
    uint b_dir_offset = uint(b_dir_offset_f);
    
    uint x_base = x_offset + b * x_stride_b;
    uint h_idx = b * hidden_size + j;
    uint h_base = b * hidden_size;
    
    uint w_h_off = w_dir_offset + (2 * hidden_size + j) * input_size;
    uint r_h_off = r_dir_offset + (2 * hidden_size + j) * hidden_size;
    
    float b_h = B[b_dir_offset + 2 * hidden_size + j] + B[b_dir_offset + 5 * hidden_size + j];
    
    // h̃_t = tanh(W_h x_t + R_h (r_t ⊙ h_{t-1}) + b_h)
    float h_acc = b_h;
    for (uint xi = x_base, wi = w_h_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
        h_acc += X[xi] * W[wi];
    }
    for (uint hi = h_base, ri = r_h_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
        h_acc += RH[hi] * R[ri];
    }
    
    // h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
    float z = Z[h_idx];
    float h_prev = H[h_idx];
    float h_new = (1.0 - z) * tanh(h_acc) + z * h_prev;
    
    H[h_idx] = h_new;
    Y[b * y_stride_b + y_offset + j] = h_new;
}
""")

        # ============================================================
        # Shader 3: Bidirectional Phase1 (linear_before_reset=0)
        # Workgroup: (batch_size, hidden_size, 1)
        # One thread processes both forward and reverse using stride offsets
        # ============================================================
        self.shader_bi_phase1 = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_x { float X[]; };
layout (binding = 1) readonly buffer buf_w { float W[]; };
layout (binding = 2) readonly buffer buf_r { float R[]; };
layout (binding = 3) readonly buffer buf_b { float B[]; };
layout (binding = 4) readonly buffer buf_h { float H[]; };      // [2, batch, hidden]
layout (binding = 5) buffer buf_rh { float RH[]; };             // [2, batch, hidden]
layout (binding = 6) buffer buf_z { float Z[]; };               // [2, batch, hidden]

layout (constant_id = 0) const float input_size_f = 0;
layout (constant_id = 1) const float hidden_size_f = 0;
layout (constant_id = 2) const float batch_hidden_f = 0;        // batch * hidden (stride for direction)
layout (constant_id = 3) const float x_stride_b_f = 0;
layout (constant_id = 4) const float x_offset_fwd_f = 0;
layout (constant_id = 5) const float x_offset_bwd_f = 0;
layout (constant_id = 6) const float w_stride_dir_f = 0;        // 3 * hidden * input
layout (constant_id = 7) const float r_stride_dir_f = 0;        // 3 * hidden * hidden
layout (constant_id = 8) const float b_stride_dir_f = 0;        // 6 * hidden

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    
    uint input_size = uint(input_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint batch_hidden = uint(batch_hidden_f);
    uint x_stride_b = uint(x_stride_b_f);
    uint x_offset_fwd = uint(x_offset_fwd_f);
    uint x_offset_bwd = uint(x_offset_bwd_f);
    uint w_stride_dir = uint(w_stride_dir_f);
    uint r_stride_dir = uint(r_stride_dir_f);
    uint b_stride_dir = uint(b_stride_dir_f);
    
    uint h_idx_local = b * hidden_size + j;
    uint h_base_local = b * hidden_size;
    
    // ========== Forward ==========
    {
        uint x_base = x_offset_fwd + b * x_stride_b;
        uint h_idx = h_idx_local;
        uint h_base = h_base_local;
        
        uint w_z_off = j * input_size;
        uint w_r_off = (hidden_size + j) * input_size;
        uint r_z_off = j * hidden_size;
        uint r_r_off = (hidden_size + j) * hidden_size;
        
        float b_z = B[j] + B[3 * hidden_size + j];
        float b_r = B[hidden_size + j] + B[4 * hidden_size + j];
        
        float z_acc = b_z;
        for (uint xi = x_base, wi = w_z_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            z_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_z_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            z_acc += H[hi] * R[ri];
        }
        float z = 1.0 / (1.0 + exp(-z_acc));
        Z[h_idx] = z;
        
        float r_acc = b_r;
        for (uint xi = x_base, wi = w_r_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            r_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_r_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            r_acc += H[hi] * R[ri];
        }
        float r = 1.0 / (1.0 + exp(-r_acc));
        RH[h_idx] = r * H[h_idx];
    }
    
    // ========== Reverse ==========
    {
        uint x_base = x_offset_bwd + b * x_stride_b;
        uint h_idx = batch_hidden + h_idx_local;
        uint h_base = batch_hidden + h_base_local;
        
        uint w_z_off = w_stride_dir + j * input_size;
        uint w_r_off = w_stride_dir + (hidden_size + j) * input_size;
        uint r_z_off = r_stride_dir + j * hidden_size;
        uint r_r_off = r_stride_dir + (hidden_size + j) * hidden_size;
        
        float b_z = B[b_stride_dir + j] + B[b_stride_dir + 3 * hidden_size + j];
        float b_r = B[b_stride_dir + hidden_size + j] + B[b_stride_dir + 4 * hidden_size + j];
        
        float z_acc = b_z;
        for (uint xi = x_base, wi = w_z_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            z_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_z_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            z_acc += H[hi] * R[ri];
        }
        float z = 1.0 / (1.0 + exp(-z_acc));
        Z[h_idx] = z;
        
        float r_acc = b_r;
        for (uint xi = x_base, wi = w_r_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            r_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_r_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            r_acc += H[hi] * R[ri];
        }
        float r = 1.0 / (1.0 + exp(-r_acc));
        RH[h_idx] = r * H[h_idx];
    }
}
""")

        # ============================================================
        # Shader 4: Bidirectional Phase2 (linear_before_reset=0)
        # Workgroup: (batch_size, hidden_size, 1)
        # One thread processes both directions, updates H, Y, Y_h
        # ============================================================
        self.shader_bi_phase2 = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_x { float X[]; };
layout (binding = 1) readonly buffer buf_w { float W[]; };
layout (binding = 2) readonly buffer buf_r { float R[]; };
layout (binding = 3) readonly buffer buf_b { float B[]; };
layout (binding = 4) buffer buf_h { float H[]; };               // [2, batch, hidden]
layout (binding = 5) readonly buffer buf_rh { float RH[]; };    // [2, batch, hidden]
layout (binding = 6) readonly buffer buf_z { float Z[]; };      // [2, batch, hidden]
layout (binding = 7) buffer buf_y { float Y[]; };
layout (binding = 8) buffer buf_yh { float Y_h[]; };            // [2, batch, hidden]

layout (constant_id = 0) const float input_size_f = 0;
layout (constant_id = 1) const float hidden_size_f = 0;
layout (constant_id = 2) const float batch_hidden_f = 0;        // batch * hidden
layout (constant_id = 3) const float x_stride_b_f = 0;
layout (constant_id = 4) const float x_offset_fwd_f = 0;
layout (constant_id = 5) const float x_offset_bwd_f = 0;
layout (constant_id = 6) const float y_stride_b_f = 0;
layout (constant_id = 7) const float y_offset_fwd_f = 0;
layout (constant_id = 8) const float y_offset_bwd_f = 0;
layout (constant_id = 9) const float w_stride_dir_f = 0;        // 3 * hidden * input
layout (constant_id = 10) const float r_stride_dir_f = 0;       // 3 * hidden * hidden
layout (constant_id = 11) const float b_stride_dir_f = 0;       // 6 * hidden

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    
    uint input_size = uint(input_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint batch_hidden = uint(batch_hidden_f);
    uint x_stride_b = uint(x_stride_b_f);
    uint x_offset_fwd = uint(x_offset_fwd_f);
    uint x_offset_bwd = uint(x_offset_bwd_f);
    uint y_stride_b = uint(y_stride_b_f);
    uint y_offset_fwd = uint(y_offset_fwd_f);
    uint y_offset_bwd = uint(y_offset_bwd_f);
    uint w_stride_dir = uint(w_stride_dir_f);
    uint r_stride_dir = uint(r_stride_dir_f);
    uint b_stride_dir = uint(b_stride_dir_f);
    
    uint h_idx_local = b * hidden_size + j;
    uint h_base_local = b * hidden_size;
    
    // ========== Forward ==========
    {
        uint x_base = x_offset_fwd + b * x_stride_b;
        uint h_idx = h_idx_local;
        uint h_base = h_base_local;
        
        uint w_h_off = (2 * hidden_size + j) * input_size;
        uint r_h_off = (2 * hidden_size + j) * hidden_size;
        float b_h = B[2 * hidden_size + j] + B[5 * hidden_size + j];
        
        float h_acc = b_h;
        for (uint xi = x_base, wi = w_h_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            h_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_h_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            h_acc += RH[hi] * R[ri];
        }
        
        float z = Z[h_idx];
        float h_prev = H[h_idx];
        float h_new = (1.0 - z) * tanh(h_acc) + z * h_prev;
        
        H[h_idx] = h_new;
        Y[b * y_stride_b + y_offset_fwd + j] = h_new;
        Y_h[h_idx] = h_new;
    }
    
    // ========== Reverse ==========
    {
        uint x_base = x_offset_bwd + b * x_stride_b;
        uint h_idx = batch_hidden + h_idx_local;
        uint h_base = batch_hidden + h_base_local;
        
        uint w_h_off = w_stride_dir + (2 * hidden_size + j) * input_size;
        uint r_h_off = r_stride_dir + (2 * hidden_size + j) * hidden_size;
        float b_h = B[b_stride_dir + 2 * hidden_size + j] + B[b_stride_dir + 5 * hidden_size + j];
        
        float h_acc = b_h;
        for (uint xi = x_base, wi = w_h_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            h_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_h_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            h_acc += RH[hi] * R[ri];
        }
        
        float z = Z[h_idx];
        float h_prev = H[h_idx];
        float h_new = (1.0 - z) * tanh(h_acc) + z * h_prev;
        
        H[h_idx] = h_new;
        Y[b * y_stride_b + y_offset_bwd + j] = h_new;
        Y_h[h_idx] = h_new;
    }
}
""")

        # ============================================================
        # Shader 5: Unidirectional Linear (linear_before_reset=1)
        # Workgroup: (batch_size, hidden_size, 1)
        # Complete GRU computation in one shader
        # ============================================================
        self.shader_uni_linear = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_x { float X[]; };
layout (binding = 1) readonly buffer buf_w { float W[]; };
layout (binding = 2) readonly buffer buf_r { float R[]; };
layout (binding = 3) readonly buffer buf_b { float B[]; };
layout (binding = 4) buffer buf_h { float H[]; };
layout (binding = 5) buffer buf_y { float Y[]; };

layout (constant_id = 0) const float input_size_f = 0;
layout (constant_id = 1) const float hidden_size_f = 0;
layout (constant_id = 2) const float x_offset_f = 0;
layout (constant_id = 3) const float x_stride_b_f = 0;
layout (constant_id = 4) const float y_offset_f = 0;
layout (constant_id = 5) const float y_stride_b_f = 0;
layout (constant_id = 6) const float w_dir_offset_f = 0;
layout (constant_id = 7) const float r_dir_offset_f = 0;
layout (constant_id = 8) const float b_dir_offset_f = 0;

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    
    uint input_size = uint(input_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint x_offset = uint(x_offset_f);
    uint x_stride_b = uint(x_stride_b_f);
    uint y_offset = uint(y_offset_f);
    uint y_stride_b = uint(y_stride_b_f);
    uint w_dir_offset = uint(w_dir_offset_f);
    uint r_dir_offset = uint(r_dir_offset_f);
    uint b_dir_offset = uint(b_dir_offset_f);
    
    uint x_base = x_offset + b * x_stride_b;
    uint h_idx = b * hidden_size + j;
    uint h_base = b * hidden_size;
    
    uint w_z_off = w_dir_offset + j * input_size;
    uint w_r_off = w_dir_offset + (hidden_size + j) * input_size;
    uint w_h_off = w_dir_offset + (2 * hidden_size + j) * input_size;
    
    uint r_z_off = r_dir_offset + j * hidden_size;
    uint r_r_off = r_dir_offset + (hidden_size + j) * hidden_size;
    uint r_h_off = r_dir_offset + (2 * hidden_size + j) * hidden_size;
    
    float b_z = B[b_dir_offset + j] + B[b_dir_offset + 3 * hidden_size + j];
    float b_r = B[b_dir_offset + hidden_size + j] + B[b_dir_offset + 4 * hidden_size + j];
    float wb_h = B[b_dir_offset + 2 * hidden_size + j];
    float rb_h = B[b_dir_offset + 5 * hidden_size + j];
    
    // z gate
    float z_acc = b_z;
    for (uint xi = x_base, wi = w_z_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
        z_acc += X[xi] * W[wi];
    }
    for (uint hi = h_base, ri = r_z_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
        z_acc += H[hi] * R[ri];
    }
    float z = 1.0 / (1.0 + exp(-z_acc));
    
    // r gate
    float r_acc = b_r;
    for (uint xi = x_base, wi = w_r_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
        r_acc += X[xi] * W[wi];
    }
    for (uint hi = h_base, ri = r_r_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
        r_acc += H[hi] * R[ri];
    }
    float r = 1.0 / (1.0 + exp(-r_acc));
    
    // h̃_t = tanh(W_h x_t + r_t ⊙ (R_h h_{t-1} + rb_h) + wb_h)
    float h_acc = wb_h;
    for (uint xi = x_base, wi = w_h_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
        h_acc += X[xi] * W[wi];
    }
    float hr_acc = rb_h;
    for (uint hi = h_base, ri = r_h_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
        hr_acc += H[hi] * R[ri];
    }
    h_acc += r * hr_acc;
    
    // h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
    float h_prev = H[h_idx];
    float h_new = (1.0 - z) * tanh(h_acc) + z * h_prev;
    
    H[h_idx] = h_new;
    Y[b * y_stride_b + y_offset + j] = h_new;
}
""")

        # ============================================================
        # Shader 6: Bidirectional Linear (linear_before_reset=1)
        # Workgroup: (batch_size, hidden_size, 1)
        # One thread processes both directions + Y_h write
        # ============================================================
        self.shader_bi_linear = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_x { float X[]; };
layout (binding = 1) readonly buffer buf_w { float W[]; };
layout (binding = 2) readonly buffer buf_r { float R[]; };
layout (binding = 3) readonly buffer buf_b { float B[]; };
layout (binding = 4) buffer buf_h { float H[]; };               // [2, batch, hidden]
layout (binding = 5) buffer buf_y { float Y[]; };
layout (binding = 6) buffer buf_yh { float Y_h[]; };            // [2, batch, hidden]

layout (constant_id = 0) const float input_size_f = 0;
layout (constant_id = 1) const float hidden_size_f = 0;
layout (constant_id = 2) const float batch_hidden_f = 0;        // batch * hidden
layout (constant_id = 3) const float x_stride_b_f = 0;
layout (constant_id = 4) const float x_offset_fwd_f = 0;
layout (constant_id = 5) const float x_offset_bwd_f = 0;
layout (constant_id = 6) const float y_stride_b_f = 0;
layout (constant_id = 7) const float y_offset_fwd_f = 0;
layout (constant_id = 8) const float y_offset_bwd_f = 0;
layout (constant_id = 9) const float w_stride_dir_f = 0;        // 3 * hidden * input
layout (constant_id = 10) const float r_stride_dir_f = 0;       // 3 * hidden * hidden
layout (constant_id = 11) const float b_stride_dir_f = 0;       // 6 * hidden

void main() {
    uint b = gl_GlobalInvocationID.x;
    uint j = gl_GlobalInvocationID.y;
    
    uint input_size = uint(input_size_f);
    uint hidden_size = uint(hidden_size_f);
    uint batch_hidden = uint(batch_hidden_f);
    uint x_stride_b = uint(x_stride_b_f);
    uint x_offset_fwd = uint(x_offset_fwd_f);
    uint x_offset_bwd = uint(x_offset_bwd_f);
    uint y_stride_b = uint(y_stride_b_f);
    uint y_offset_fwd = uint(y_offset_fwd_f);
    uint y_offset_bwd = uint(y_offset_bwd_f);
    uint w_stride_dir = uint(w_stride_dir_f);
    uint r_stride_dir = uint(r_stride_dir_f);
    uint b_stride_dir = uint(b_stride_dir_f);
    
    uint h_idx_local = b * hidden_size + j;
    uint h_base_local = b * hidden_size;
    
    // ========== Forward ==========
    {
        uint x_base = x_offset_fwd + b * x_stride_b;
        uint h_idx = h_idx_local;
        uint h_base = h_base_local;
        
        uint w_z_off = j * input_size;
        uint w_r_off = (hidden_size + j) * input_size;
        uint w_h_off = (2 * hidden_size + j) * input_size;
        uint r_z_off = j * hidden_size;
        uint r_r_off = (hidden_size + j) * hidden_size;
        uint r_h_off = (2 * hidden_size + j) * hidden_size;
        
        float b_z = B[j] + B[3 * hidden_size + j];
        float b_r = B[hidden_size + j] + B[4 * hidden_size + j];
        float wb_h = B[2 * hidden_size + j];
        float rb_h = B[5 * hidden_size + j];
        
        float z_acc = b_z;
        for (uint xi = x_base, wi = w_z_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            z_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_z_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            z_acc += H[hi] * R[ri];
        }
        float z = 1.0 / (1.0 + exp(-z_acc));
        
        float r_acc = b_r;
        for (uint xi = x_base, wi = w_r_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            r_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_r_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            r_acc += H[hi] * R[ri];
        }
        float r = 1.0 / (1.0 + exp(-r_acc));
        
        float h_acc = wb_h;
        for (uint xi = x_base, wi = w_h_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            h_acc += X[xi] * W[wi];
        }
        float hr_acc = rb_h;
        for (uint hi = h_base, ri = r_h_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            hr_acc += H[hi] * R[ri];
        }
        h_acc += r * hr_acc;
        
        float h_prev = H[h_idx];
        float h_new = (1.0 - z) * tanh(h_acc) + z * h_prev;
        
        H[h_idx] = h_new;
        Y[b * y_stride_b + y_offset_fwd + j] = h_new;
        Y_h[h_idx] = h_new;
    }
    
    // ========== Reverse ==========
    {
        uint x_base = x_offset_bwd + b * x_stride_b;
        uint h_idx = batch_hidden + h_idx_local;
        uint h_base = batch_hidden + h_base_local;
        
        uint w_z_off = w_stride_dir + j * input_size;
        uint w_r_off = w_stride_dir + (hidden_size + j) * input_size;
        uint w_h_off = w_stride_dir + (2 * hidden_size + j) * input_size;
        uint r_z_off = r_stride_dir + j * hidden_size;
        uint r_r_off = r_stride_dir + (hidden_size + j) * hidden_size;
        uint r_h_off = r_stride_dir + (2 * hidden_size + j) * hidden_size;
        
        float b_z = B[b_stride_dir + j] + B[b_stride_dir + 3 * hidden_size + j];
        float b_r = B[b_stride_dir + hidden_size + j] + B[b_stride_dir + 4 * hidden_size + j];
        float wb_h = B[b_stride_dir + 2 * hidden_size + j];
        float rb_h = B[b_stride_dir + 5 * hidden_size + j];
        
        float z_acc = b_z;
        for (uint xi = x_base, wi = w_z_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            z_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_z_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            z_acc += H[hi] * R[ri];
        }
        float z = 1.0 / (1.0 + exp(-z_acc));
        
        float r_acc = b_r;
        for (uint xi = x_base, wi = w_r_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            r_acc += X[xi] * W[wi];
        }
        for (uint hi = h_base, ri = r_r_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            r_acc += H[hi] * R[ri];
        }
        float r = 1.0 / (1.0 + exp(-r_acc));
        
        float h_acc = wb_h;
        for (uint xi = x_base, wi = w_h_off, cnt = 0; cnt < input_size; ++xi, ++wi, ++cnt) {
            h_acc += X[xi] * W[wi];
        }
        float hr_acc = rb_h;
        for (uint hi = h_base, ri = r_h_off, cnt = 0; cnt < hidden_size; ++hi, ++ri, ++cnt) {
            hr_acc += H[hi] * R[ri];
        }
        h_acc += r * hr_acc;
        
        float h_prev = H[h_idx];
        float h_new = (1.0 - z) * tanh(h_acc) + z * h_prev;
        
        H[h_idx] = h_new;
        Y[b * y_stride_b + y_offset_bwd + j] = h_new;
        Y_h[h_idx] = h_new;
    }
}
""")

    def __repr__(self):
        return f"GruOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp is None:
                input_tensors.append((None, []))
            else:
                numpy_in = inp.reshape(-1).astype(np.float32)
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors if t[0] is not None] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor for tensor, _ in output_tensor_and_shape]))
        seq.eval()

        output_list = []
        for tensor, shape_out in output_tensor_and_shape:
            output_list.append(tensor.data().reshape(shape_out))

        for tensor, _ in input_tensors:
            if tensor is not None:
                del tensor
        del updated_tensors
        return output_list

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_X, shape_X = input_tensors[0]
        tensor_W, shape_W = input_tensors[1]
        tensor_R, shape_R = input_tensors[2]
        
        num_directions = shape_W[0]
        hidden_size = self.hidden_size if self.hidden_size else shape_R[-1]
        input_size = shape_W[2]
        
        # Validate direction and num_directions consistency
        is_bidirectional = self.direction == "bidirectional"
        if is_bidirectional:
            assert num_directions == 2, f"bidirectional requires num_directions=2, got {num_directions}"
        else:
            assert num_directions == 1, f"forward/reverse requires num_directions=1, got {num_directions}"
        
        # Handle layout
        if self.layout == 1:
            batch_size = shape_X[0]
            seq_length = shape_X[1]
        else:
            seq_length = shape_X[0]
            batch_size = shape_X[1]
        
        # Output tensor Y: shape includes num_directions
        # layout=0: [seq_length, num_directions, batch_size, hidden_size]
        # layout=1: [batch_size, seq_length, num_directions, hidden_size]
        tensor_Y = self.manager.tensor(np.zeros(seq_length * num_directions * batch_size * hidden_size, dtype=np.float32))
        updated_tensors.append(tensor_Y)
        
        # X layout: layout=0 -> [seq, batch, input], layout=1 -> [batch, seq, input]
        if self.layout == 0:
            x_stride_b = input_size
            x_offset_base = batch_size * input_size
        else:
            x_stride_b = seq_length * input_size
            x_offset_base = input_size
        
        # Y stride and offset base calculation
        if self.layout == 0:
            y_stride_b = hidden_size
            y_offset_base = num_directions * batch_size * hidden_size
        else:
            y_stride_b = seq_length * num_directions * hidden_size
            y_offset_base = num_directions * hidden_size
        
        # Handle optional B - create zero tensor if not provided
        if len(input_tensors) > 3 and input_tensors[3][0] is not None:
            tensor_B = input_tensors[3][0]
        else:
            tensor_B = self.manager.tensor(np.zeros(num_directions * 2 * self.number_of_gates * hidden_size, dtype=np.float32))
            updated_tensors.append(tensor_B)
        
        # ========== BIDIRECTIONAL PATH ==========
        if is_bidirectional:
            # For bidirectional: use unified H tensor [2, batch, hidden]
            batch_hidden = batch_size * hidden_size
            
            if len(input_tensors) > 5 and input_tensors[5][0] is not None:
                # initial_h provided: [2, batch, hidden]
                shape_H = input_tensors[5][1]
                H_np = input_tensors[5][0].data().reshape(shape_H)
                tensor_H = self.manager.tensor(H_np.reshape(-1).astype(np.float32))
            else:
                tensor_H = self.manager.tensor(np.zeros(2 * batch_hidden, dtype=np.float32))
            updated_tensors.append(tensor_H)
            
            # Y_h tensor: [2, batch, hidden] - written each step, final step is the result
            tensor_Y_h = self.manager.tensor(np.zeros(2 * batch_hidden, dtype=np.float32))
            updated_tensors.append(tensor_Y_h)
            
            # Pre-compute stride constants for direction offsets
            w_stride_dir = 3 * hidden_size * input_size
            r_stride_dir = 3 * hidden_size * hidden_size
            b_stride_dir = 6 * hidden_size
            
            if self.linear_before_reset:
                # Bidirectional + linear_before_reset=1: shader_bi_linear
                for step in range(seq_length):
                    t_fwd = step
                    t_bwd = seq_length - 1 - step
                    
                    x_offset_fwd = t_fwd * x_offset_base
                    x_offset_bwd = t_bwd * x_offset_base
                    
                    # Y offsets for each direction
                    y_offset_fwd = t_fwd * y_offset_base + 0 * (batch_size * hidden_size if self.layout == 0 else hidden_size)
                    y_offset_bwd = t_bwd * y_offset_base + 1 * (batch_size * hidden_size if self.layout == 0 else hidden_size)
                    
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor_X, tensor_W, tensor_R, tensor_B, tensor_H, tensor_Y, tensor_Y_h],
                        self.shader_bi_linear,
                        (batch_size, hidden_size, 1),  # One thread handles both directions
                        [input_size, hidden_size, batch_hidden, x_stride_b,
                         x_offset_fwd, x_offset_bwd, y_stride_b, y_offset_fwd, y_offset_bwd,
                         w_stride_dir, r_stride_dir, b_stride_dir],
                        []
                    ))
            else:
                # Bidirectional + linear_before_reset=0: shader_bi_phase1 + shader_bi_phase2
                tensor_RH = self.manager.tensor(np.zeros(2 * batch_hidden, dtype=np.float32))
                tensor_Z = self.manager.tensor(np.zeros(2 * batch_hidden, dtype=np.float32))
                updated_tensors.append(tensor_RH)
                updated_tensors.append(tensor_Z)
                
                for step in range(seq_length):
                    t_fwd = step
                    t_bwd = seq_length - 1 - step
                    
                    x_offset_fwd = t_fwd * x_offset_base
                    x_offset_bwd = t_bwd * x_offset_base
                    
                    y_offset_fwd = t_fwd * y_offset_base + 0 * (batch_size * hidden_size if self.layout == 0 else hidden_size)
                    y_offset_bwd = t_bwd * y_offset_base + 1 * (batch_size * hidden_size if self.layout == 0 else hidden_size)
                    
                    # Phase 1
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor_X, tensor_W, tensor_R, tensor_B, tensor_H, tensor_RH, tensor_Z],
                        self.shader_bi_phase1,
                        (batch_size, hidden_size, 1),  # One thread handles both directions
                        [input_size, hidden_size, batch_hidden, x_stride_b, x_offset_fwd, x_offset_bwd,
                         w_stride_dir, r_stride_dir, b_stride_dir],
                        []
                    ))
                    # Phase 2
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor_X, tensor_W, tensor_R, tensor_B, tensor_H, tensor_RH, tensor_Z, tensor_Y, tensor_Y_h],
                        self.shader_bi_phase2,
                        (batch_size, hidden_size, 1),  # One thread handles both directions
                        [input_size, hidden_size, batch_hidden, x_stride_b,
                         x_offset_fwd, x_offset_bwd, y_stride_b, y_offset_fwd, y_offset_bwd,
                         w_stride_dir, r_stride_dir, b_stride_dir],
                        []
                    ))
            
            # Output shapes
            if self.layout == 1:
                output_shape_Y = [batch_size, seq_length, num_directions, hidden_size]
            else:
                output_shape_Y = [seq_length, num_directions, batch_size, hidden_size]
            output_shape_Y_h = [num_directions, batch_size, hidden_size]
            
            return [(tensor_Y, output_shape_Y), (tensor_Y_h, output_shape_Y_h)]
        
        # ========== UNIDIRECTIONAL PATH (forward or reverse) ==========
        else:
            is_reverse = self.direction == "reverse"
            
            # H tensor for single direction: [batch, hidden]
            if len(input_tensors) > 5 and input_tensors[5][0] is not None:
                shape_H = input_tensors[5][1]
                H_np = input_tensors[5][0].data().reshape(shape_H)
                # For unidirectional, initial_h is [1, batch, hidden], take [0]
                tensor_H = self.manager.tensor(H_np[0].reshape(-1).astype(np.float32))
            else:
                tensor_H = self.manager.tensor(np.zeros(batch_size * hidden_size, dtype=np.float32))
            updated_tensors.append(tensor_H)
            
            # Weight/bias offsets (always 0 for unidirectional)
            w_dir_offset = 0
            r_dir_offset = 0
            b_dir_offset = 0
            
            if self.linear_before_reset:
                # Unidirectional + linear_before_reset=1: shader_uni_linear
                for step in range(seq_length):
                    t = (seq_length - 1 - step) if is_reverse else step
                    
                    x_offset = t * x_offset_base
                    y_offset = t * y_offset_base
                    
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor_X, tensor_W, tensor_R, tensor_B, tensor_H, tensor_Y],
                        self.shader_uni_linear,
                        (batch_size, hidden_size, 1),
                        [input_size, hidden_size, x_offset, x_stride_b, y_offset, y_stride_b,
                         w_dir_offset, r_dir_offset, b_dir_offset],
                        []
                    ))
            else:
                # Unidirectional + linear_before_reset=0: shader_uni_phase1 + shader_uni_phase2
                tensor_RH = self.manager.tensor(np.zeros(batch_size * hidden_size, dtype=np.float32))
                tensor_Z = self.manager.tensor(np.zeros(batch_size * hidden_size, dtype=np.float32))
                updated_tensors.append(tensor_RH)
                updated_tensors.append(tensor_Z)
                
                for step in range(seq_length):
                    t = (seq_length - 1 - step) if is_reverse else step
                    
                    x_offset = t * x_offset_base
                    y_offset = t * y_offset_base
                    
                    # Phase 1
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor_X, tensor_W, tensor_R, tensor_B, tensor_H, tensor_RH, tensor_Z],
                        self.shader_uni_phase1,
                        (batch_size, hidden_size, 1),
                        [input_size, hidden_size, x_offset, x_stride_b, w_dir_offset, r_dir_offset, b_dir_offset],
                        []
                    ))
                    # Phase 2
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor_X, tensor_W, tensor_R, tensor_B, tensor_H, tensor_RH, tensor_Z, tensor_Y],
                        self.shader_uni_phase2,
                        (batch_size, hidden_size, 1),
                        [input_size, hidden_size, x_offset, x_stride_b, y_offset, y_stride_b,
                         w_dir_offset, r_dir_offset, b_dir_offset],
                        []
                    ))
            
            # Output shapes for unidirectional
            if self.layout == 1:
                output_shape_Y = [batch_size, seq_length, num_directions, hidden_size]
                output_shape_Y_h = [batch_size, seq_length, hidden_size]
                return [(tensor_Y, output_shape_Y), (tensor_Y, output_shape_Y_h)]
            else:
                output_shape_Y = [seq_length, num_directions, batch_size, hidden_size]
                output_shape_Y_h = [num_directions, batch_size, hidden_size]
                return [(tensor_Y, output_shape_Y), (tensor_H, output_shape_Y_h)]
