import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class LstmOp:
    def __init__(self, manager: kp.Manager, direction="forward", hidden_size=None, activations=None, layout=0,
                 activation_alpha=None, activation_beta=None, clip=None, input_forget=0):
        self.manager = manager
        self._step_shaders = {}

        # matmul shader（layout 0：X 形状 [seq,batch,input]）
        self.matmul_shader_layout0 = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InA   {{ float A[];   }};
layout(std430, set = 0, binding = 1) readonly  buffer InB   {{ float B[];   }};
layout(std430, set = 0, binding = 2) writeonly buffer OutC  {{ float C[];   }};
layout(std430, set = 0, binding = 3) readonly  buffer Params {{
    uint seq_len;
    uint batch_size;
    uint input_size;
    uint hidden_size_4;
    uint num_directions;
    uint direction_index;
}};

void main() {{
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint s = gl_GlobalInvocationID.z;

    if (b >= batch_size || h >= hidden_size_4 || s >= seq_len) return;

    uint d = direction_index;
    float sum = 0.0;
    uint sb    = s * batch_size + b;
    uint a_idx = sb * input_size;
    uint b_idx = (d * hidden_size_4 + h) * input_size;
    for (uint k = 0u; k < input_size; k++, a_idx++, b_idx++) {{
        sum += A[a_idx] * B[b_idx];
    }}
    uint c_idx = (sb * num_directions + d) * hidden_size_4 + h;
    C[c_idx] = sum;
}}
""")

        # matmul shader（layout 1：X 形状 [batch,seq,input]）
        self.matmul_shader_layout1 = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InA   {{ float A[];   }};
layout(std430, set = 0, binding = 1) readonly  buffer InB   {{ float B[];   }};
layout(std430, set = 0, binding = 2) writeonly buffer OutC  {{ float C[];   }};
layout(std430, set = 0, binding = 3) readonly  buffer Params {{
    uint seq_len;
    uint batch_size;
    uint input_size;
    uint hidden_size_4;
    uint num_directions;
    uint direction_index;
}};

void main() {{
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint s = gl_GlobalInvocationID.z;

    if (b >= batch_size || h >= hidden_size_4 || s >= seq_len) return;

    uint d = direction_index;
    float sum = 0.0;
    uint a_idx = (b * seq_len + s) * input_size;
    uint b_idx = (d * hidden_size_4 + h) * input_size;
    for (uint k = 0u; k < input_size; k++, a_idx++, b_idx++) {{
        sum += A[a_idx] * B[b_idx];
    }}
    uint sb    = s * batch_size + b;
    uint c_idx = (sb * num_directions + d) * hidden_size_4 + h;
    C[c_idx] = sum;
}}
""")

        self.set_attributes(
            direction=direction,
            hidden_size=hidden_size,
            activations=activations,
            layout=layout,
            activation_alpha=activation_alpha,
            activation_beta=activation_beta,
            clip=clip,
            input_forget=input_forget
        )

    def set_attributes(self, direction=None, layout=None, hidden_size=None, activations=None,
                       activation_alpha=None, activation_beta=None, clip=None, input_forget=None):
        if direction is not None:
            assert direction in ['forward', 'reverse', 'bidirectional'], f"Invalid direction: {direction}"
            self.direction = direction
        if layout is not None:
            assert layout in [0, 1], f"Invalid layout: {layout}"
            self.layout = layout
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if activations is not None:
            self.activations = activations
        else:
            self.activations = ["Sigmoid", "Tanh", "Tanh"]
        if activation_alpha is not None:
            self.activation_alpha = activation_alpha
        else:
            self.activation_alpha = []
        if activation_beta is not None:
            self.activation_beta = activation_beta
        else:
            self.activation_beta = []
        if clip is not None:
            self.clip = clip
        elif not hasattr(self, 'clip'):
            self.clip = None
        if input_forget is not None:
            self.input_forget = input_forget

        self._setup_activation_params()
        self._step_shaders = {}
        self._generate_all_step_shaders()

    def _generate_all_step_shaders(self):
        if self.direction == "bidirectional":
            for has_bias in [True, False]:
                for has_seq_lens in [True, False]:
                    for has_peepholes in [True, False]:
                        key = (self.direction, self.layout, has_bias, has_seq_lens, has_peepholes)
                        self._step_shaders[key] = self._generate_bidirectional_shader(
                            self.layout, has_bias, has_seq_lens, has_peepholes
                        )
        else:
            for has_bias in [True, False]:
                for has_seq_lens in [True, False]:
                    for has_peepholes in [True, False]:
                        key = (self.direction, self.layout, has_bias, has_seq_lens, has_peepholes)
                        self._step_shaders[key] = self._generate_unidirectional_shader(
                            self.direction, self.layout, has_bias, has_seq_lens, has_peepholes
                        )

    def _setup_activation_params(self):
        num_directions = 2 if self.direction == "bidirectional" else 1
        alpha_idx = 0
        beta_idx  = 0

        self.act_f_0_name  = self.activations[0].lower()
        self.act_f_0_alpha = self.activation_alpha[alpha_idx] if alpha_idx < len(self.activation_alpha) else 0.0; alpha_idx += 1
        self.act_f_0_beta  = self.activation_beta[beta_idx]   if beta_idx  < len(self.activation_beta)  else 0.0; beta_idx  += 1

        self.act_g_0_name  = self.activations[1].lower()
        self.act_g_0_alpha = self.activation_alpha[alpha_idx] if alpha_idx < len(self.activation_alpha) else 0.0; alpha_idx += 1
        self.act_g_0_beta  = self.activation_beta[beta_idx]   if beta_idx  < len(self.activation_beta)  else 0.0; beta_idx  += 1

        self.act_h_0_name  = self.activations[2].lower()
        self.act_h_0_alpha = self.activation_alpha[alpha_idx] if alpha_idx < len(self.activation_alpha) else 0.0; alpha_idx += 1
        self.act_h_0_beta  = self.activation_beta[beta_idx]   if beta_idx  < len(self.activation_beta)  else 0.0; beta_idx  += 1

        if num_directions == 2:
            self.act_f_1_name  = self.activations[3].lower()
            self.act_f_1_alpha = self.activation_alpha[alpha_idx] if alpha_idx < len(self.activation_alpha) else 0.0; alpha_idx += 1
            self.act_f_1_beta  = self.activation_beta[beta_idx]   if beta_idx  < len(self.activation_beta)  else 0.0; beta_idx  += 1

            self.act_g_1_name  = self.activations[4].lower()
            self.act_g_1_alpha = self.activation_alpha[alpha_idx] if alpha_idx < len(self.activation_alpha) else 0.0; alpha_idx += 1
            self.act_g_1_beta  = self.activation_beta[beta_idx]   if beta_idx  < len(self.activation_beta)  else 0.0; beta_idx  += 1

            self.act_h_1_name  = self.activations[5].lower()
            self.act_h_1_alpha = self.activation_alpha[alpha_idx] if alpha_idx < len(self.activation_alpha) else 0.0; alpha_idx += 1
            self.act_h_1_beta  = self.activation_beta[beta_idx]   if beta_idx  < len(self.activation_beta)  else 0.0; beta_idx  += 1

    def _generate_activation_function(self, activation_name, alpha, beta, func_name):
        activation_impls = {
            'tanh':           'return tanh(val);',
            'relu':           'return max(val, 0.0);',
            'leakyrelu':      'return (val >= 0.0) ? val : (val * alpha);',
            'thresholdedrelu':'return (val > alpha) ? val : 0.0;',
            'sigmoid':        'return 1.0 / (1.0 + exp(-val));',
            'hardsigmoid':    'return max(0.0, min(1.0, alpha * val + beta));',
            'elu':            'return (val > 0.0) ? val : alpha * (exp(val) - 1.0);',
            'softsign':       'return val / (1.0 + abs(val));',
            'softplus':       'return log(1.0 + exp(val));',
            'scaledtanh':     'return alpha * tanh(beta * val);',
            'affine':         'return alpha * val + beta;',
        }
        act_impl = activation_impls.get(activation_name, 'return tanh(val);')
        return f"""
float {func_name}(float val, float alpha, float beta) {{
    {act_impl}
}}"""

    def _generate_unidirectional_shader(self, direction, layout_mode, has_bias, has_seq_lens, has_peepholes):
        activation_code  = self._generate_activation_function(self.act_f_0_name, self.act_f_0_alpha, self.act_f_0_beta, "activation_f")
        activation_code += self._generate_activation_function(self.act_g_0_name, self.act_g_0_alpha, self.act_g_0_beta, "activation_g")
        activation_code += self._generate_activation_function(self.act_h_0_name, self.act_h_0_alpha, self.act_h_0_beta, "activation_h")

        if direction == "forward":
            t_calc      = "uint t      = step_index;"
            t_prev_calc = "uint t_prev = t - 1u;"
        else:
            t_calc      = "uint t      = seq_len - 1u - step_index;"
            t_prev_calc = "uint t_prev = t + 1u;"

        if layout_mode == 0:
            y_idx_calc      = "uint y_idx      = t * batch * hidden + b * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = t_prev * batch * hidden + b * hidden;"
            xw_idx_calc     = "uint xw_base    = t * batch * hidden_4 + b * hidden_4;"
        else:
            y_idx_calc      = "uint y_idx      = b * seq_len * hidden + t * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = b * seq_len * hidden + t_prev * hidden;"
            xw_idx_calc     = "uint xw_base    = t * batch * hidden_4 + b * hidden_4;"

        seq_len_val_decl = "uint seq_len_val = SeqLens[b];" if has_seq_lens else "uint seq_len_val = seq_len;"

        bias_logic = """
    uint b_offset = 0u;
    float b_wi = Bias[b_offset + h];
    float b_wo = Bias[b_offset + h + hidden];
    float b_wf = Bias[b_offset + h + hidden * 2u];
    float b_wc = Bias[b_offset + h + hidden * 3u];
    float b_ri = Bias[b_offset + hidden_4 + h];
    float b_ro = Bias[b_offset + hidden_4 + h + hidden];
    float b_rf = Bias[b_offset + hidden_4 + h + hidden * 2u];
    float b_rc = Bias[b_offset + hidden_4 + h + hidden * 3u];""" if has_bias else """
    float b_wi = 0.0, b_wo = 0.0, b_wf = 0.0, b_wc = 0.0;
    float b_ri = 0.0, b_ro = 0.0, b_rf = 0.0, b_rc = 0.0;"""

        peephole_i = "it += P[h]           * ct_prev;" if has_peepholes else ""
        peephole_f = "ft += P[hidden + h]  * ct_prev;" if has_peepholes else ""
        peephole_o = "ot += P[hidden*2u+h] * ct;"      if has_peepholes else ""

        clip_logic = """
    it       = clamp(it,       -clip_val, clip_val);
    ot       = clamp(ot,       -clip_val, clip_val);
    ft       = clamp(ft,       -clip_val, clip_val);
    ct_tilde = clamp(ct_tilde, -clip_val, clip_val);""" if self.clip else ""

        if direction == "forward":
            last_step_cond = "t == seq_len_val - 1u"
        else:
            last_step_cond = "t == 0u"

        shader_source = f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout(std430, set = 0, binding =  0) readonly  buffer InXW     {{ float XW[];     }};
layout(std430, set = 0, binding =  1) readonly  buffer InHInit  {{ float H_init[]; }};
layout(std430, set = 0, binding =  2) readonly  buffer InCInit  {{ float C_init[]; }};
layout(std430, set = 0, binding =  3) readonly  buffer InR      {{ float R[];      }};
layout(std430, set = 0, binding =  4) readonly  buffer InBias   {{ float Bias[];   }};
layout(std430, set = 0, binding =  5) readonly  buffer InP      {{ float P[];      }};
layout(std430, set = 0, binding =  6) writeonly buffer OutY     {{ float Y_curr[]; }};
layout(std430, set = 0, binding =  7) readonly  buffer InY      {{ float Y_prev[]; }};
layout(std430, set = 0, binding =  8) writeonly buffer OutC     {{ float C_curr[]; }};
layout(std430, set = 0, binding =  9) readonly  buffer InC      {{ float C_prev[]; }};
layout(std430, set = 0, binding = 10) writeonly buffer OutYh    {{ float Yh[];     }};
layout(std430, set = 0, binding = 11) writeonly buffer OutCh    {{ float C_h[];    }};
layout(std430, set = 0, binding = 12) readonly  buffer InSeqLens{{ uint  SeqLens[];}};
layout(std430, set = 0, binding = 13) readonly  buffer Params   {{
    uint  batch;
    uint  hidden;
    uint  seq_len;
    uint  step_index;
    float clip_val;
    float input_forget_f;
}};

{activation_code}

void main() {{
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;

    if (b >= batch || h >= hidden) return;

    uint hidden_4 = hidden * 4u;
    float f_alpha = {self.act_f_0_alpha};
    float f_beta  = {self.act_f_0_beta};
    float g_alpha = {self.act_g_0_alpha};
    float g_beta  = {self.act_g_0_beta};
    float h_alpha = {self.act_h_0_alpha};
    float h_beta  = {self.act_h_0_beta};

    {t_calc}
    {t_prev_calc}
    {seq_len_val_decl}

    bool is_padding = (t >= seq_len_val);
    float ht = 0.0, ct = 0.0;

    if (!is_padding) {{
        {xw_idx_calc}
        float xw_i = XW[xw_base + h];
        float xw_o = XW[xw_base + h + hidden];
        float xw_f = XW[xw_base + h + hidden * 2u];
        float xw_c = XW[xw_base + h + hidden * 3u];

        uint b_mul_hidden = b * hidden;
        float r_i = 0.0, r_o = 0.0, r_f = 0.0, r_c = 0.0;
        float ct_prev = 0.0;

        if (step_index == 0u) {{
            uint h_idx = b_mul_hidden;
            for (uint k = 0u; k < hidden; k++) {{
                float hv    = H_init[h_idx];
                uint r_base = k * hidden + h;
                r_i += hv * R[r_base];
                r_o += hv * R[r_base + hidden * hidden];
                r_f += hv * R[r_base + hidden * hidden * 2u];
                r_c += hv * R[r_base + hidden * hidden * 3u];
                h_idx++;
            }}
            ct_prev = C_init[b_mul_hidden + h];
        }} else {{
            {y_prev_idx_calc}
            for (uint k = 0u; k < hidden; k++) {{
                float hv    = Y_prev[y_prev_idx + k];
                uint r_base = k * hidden + h;
                r_i += hv * R[r_base];
                r_o += hv * R[r_base + hidden * hidden];
                r_f += hv * R[r_base + hidden * hidden * 2u];
                r_c += hv * R[r_base + hidden * hidden * 3u];
            }}
            ct_prev = C_prev[y_prev_idx + h];
        }}

        {bias_logic}

        float it       = xw_i + r_i + b_wi + b_ri;
        float ft       = xw_f + r_f + b_wf + b_rf;
        float ct_tilde = xw_c + r_c + b_wc + b_rc;
        float ot       = xw_o + r_o + b_wo + b_ro;

        {peephole_i}
        {peephole_f}
        {clip_logic}

        it       = activation_f(it,       f_alpha, f_beta);
        ft       = activation_f(ft,       f_alpha, f_beta);
        ct_tilde = activation_g(ct_tilde, g_alpha, g_beta);
        ct       = ft * ct_prev + it * ct_tilde;

        {peephole_o}
        ot = activation_f(ot, f_alpha, f_beta);
        ht = ot * activation_h(ct, h_alpha, h_beta);
    }}

    {y_idx_calc}
    Y_curr[y_idx] = ht;
    C_curr[y_idx] = ct;

    if (seq_len_val > 0u) {{
        if ({last_step_cond}) {{
            Yh[b * hidden + h] = ht;
            C_h[b * hidden + h] = ct;
        }}
    }} else {{
        if (step_index == 0u) {{
            Yh[b * hidden + h] = H_init[b * hidden + h];
            C_h[b * hidden + h] = C_init[b * hidden + h];
        }}
    }}
}}
"""
        return compile_source(shader_source)

    def _generate_bidirectional_shader(self, layout_mode, has_bias, has_seq_lens, has_peepholes):
        activation_code  = self._generate_activation_function(self.act_f_0_name, self.act_f_0_alpha, self.act_f_0_beta, "activation_f_0")
        activation_code += self._generate_activation_function(self.act_g_0_name, self.act_g_0_alpha, self.act_g_0_beta, "activation_g_0")
        activation_code += self._generate_activation_function(self.act_h_0_name, self.act_h_0_alpha, self.act_h_0_beta, "activation_h_0")
        activation_code += self._generate_activation_function(self.act_f_1_name, self.act_f_1_alpha, self.act_f_1_beta, "activation_f_1")
        activation_code += self._generate_activation_function(self.act_g_1_name, self.act_g_1_alpha, self.act_g_1_beta, "activation_g_1")
        activation_code += self._generate_activation_function(self.act_h_1_name, self.act_h_1_alpha, self.act_h_1_beta, "activation_h_1")

        if layout_mode == 0:
            y_idx_calc      = "uint y_idx      = t * 2u * batch * hidden + d * batch * hidden + b * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = t_prev * 2u * batch * hidden + d * batch * hidden + b * hidden;"
            xw_idx_calc     = "uint xw_base    = ((t * batch + b) * 2u + d) * hidden_4;"
        else:
            y_idx_calc      = "uint y_idx      = b * seq_len * 2u * hidden + t * 2u * hidden + d * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = b * seq_len * 2u * hidden + t_prev * 2u * hidden + d * hidden;"
            xw_idx_calc     = "uint xw_base    = ((t * batch + b) * 2u + d) * hidden_4;"

        seq_len_val_decl = "uint seq_len_val = SeqLens[b];" if has_seq_lens else "uint seq_len_val = seq_len;"

        bias_logic = """
        uint b_offset = d * 8u * hidden;
        float b_wi = Bias[b_offset + h];
        float b_wo = Bias[b_offset + h + hidden];
        float b_wf = Bias[b_offset + h + hidden * 2u];
        float b_wc = Bias[b_offset + h + hidden * 3u];
        float b_ri = Bias[b_offset + hidden_4 + h];
        float b_ro = Bias[b_offset + hidden_4 + h + hidden];
        float b_rf = Bias[b_offset + hidden_4 + h + hidden * 2u];
        float b_rc = Bias[b_offset + hidden_4 + h + hidden * 3u];""" if has_bias else """
        float b_wi = 0.0, b_wo = 0.0, b_wf = 0.0, b_wc = 0.0;
        float b_ri = 0.0, b_ro = 0.0, b_rf = 0.0, b_rc = 0.0;"""

        peephole_i = "it += P[d * 3u * hidden + h]           * ct_prev;" if has_peepholes else ""
        peephole_f = "ft += P[d * 3u * hidden + hidden + h]  * ct_prev;" if has_peepholes else ""
        peephole_o = "ot += P[d * 3u * hidden + hidden*2u+h] * ct;"      if has_peepholes else ""

        clip_logic = """
        it       = clamp(it,       -clip_val, clip_val);
        ot       = clamp(ot,       -clip_val, clip_val);
        ft       = clamp(ft,       -clip_val, clip_val);
        ct_tilde = clamp(ct_tilde, -clip_val, clip_val);""" if self.clip else ""

        shader_source = f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding =  0) readonly  buffer InXW     {{ float XW[];     }};
layout(std430, set = 0, binding =  1) readonly  buffer InHInit  {{ float H_init[]; }};
layout(std430, set = 0, binding =  2) readonly  buffer InCInit  {{ float C_init[]; }};
layout(std430, set = 0, binding =  3) readonly  buffer InR      {{ float R[];      }};
layout(std430, set = 0, binding =  4) readonly  buffer InBias   {{ float Bias[];   }};
layout(std430, set = 0, binding =  5) readonly  buffer InP      {{ float P[];      }};
layout(std430, set = 0, binding =  6) writeonly buffer OutY     {{ float Y_curr[]; }};
layout(std430, set = 0, binding =  7) readonly  buffer InY      {{ float Y_prev[]; }};
layout(std430, set = 0, binding =  8) writeonly buffer OutC     {{ float C_curr[]; }};
layout(std430, set = 0, binding =  9) readonly  buffer InC      {{ float C_prev[]; }};
layout(std430, set = 0, binding = 10) writeonly buffer OutYh    {{ float Yh[];     }};
layout(std430, set = 0, binding = 11) writeonly buffer OutCh    {{ float C_h[];    }};
layout(std430, set = 0, binding = 12) readonly  buffer InSeqLens{{ uint  SeqLens[];}};
layout(std430, set = 0, binding = 13) readonly  buffer Params   {{
    uint  batch;
    uint  hidden;
    uint  seq_len;
    uint  step_index;
    float clip_val;
    float input_forget_f;
}};

{activation_code}

void main() {{
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint d = gl_GlobalInvocationID.z;

    if (b >= batch || h >= hidden || d >= 2u) return;

    uint hidden_4 = hidden * 4u;
    float f_alpha_0 = {self.act_f_0_alpha};
    float f_beta_0  = {self.act_f_0_beta};
    float g_alpha_0 = {self.act_g_0_alpha};
    float g_beta_0  = {self.act_g_0_beta};
    float h_alpha_0 = {self.act_h_0_alpha};
    float h_beta_0  = {self.act_h_0_beta};
    float f_alpha_1 = {self.act_f_1_alpha};
    float f_beta_1  = {self.act_f_1_beta};
    float g_alpha_1 = {self.act_g_1_alpha};
    float g_beta_1  = {self.act_g_1_beta};
    float h_alpha_1 = {self.act_h_1_alpha};
    float h_beta_1  = {self.act_h_1_beta};

    bool is_reverse = (d == 1u);
    uint t      = is_reverse ? (seq_len - 1u - step_index) : step_index;
    uint t_prev = is_reverse ? (t + 1u) : (t - 1u);

    {seq_len_val_decl}

    bool is_padding = (t >= seq_len_val);
    float ht = 0.0, ct = 0.0;

    if (!is_padding) {{
        {xw_idx_calc}
        float xw_i = XW[xw_base + h];
        float xw_o = XW[xw_base + h + hidden];
        float xw_f = XW[xw_base + h + hidden * 2u];
        float xw_c = XW[xw_base + h + hidden * 3u];

        uint d_mul_batch_hidden  = d * batch * hidden;
        uint b_mul_hidden        = b * hidden;
        uint d_mul_h4_hidden     = d * hidden_4 * hidden;

        float r_i = 0.0, r_o = 0.0, r_f = 0.0, r_c = 0.0;
        float ct_prev = 0.0;

        if (step_index == 0u) {{
            uint h_idx = b_mul_hidden;
            for (uint k = 0u; k < hidden; k++) {{
                float hv    = H_init[h_idx];
                uint r_base = d_mul_h4_hidden + k * hidden + h;
                r_i += hv * R[r_base];
                r_o += hv * R[r_base + hidden * hidden];
                r_f += hv * R[r_base + hidden * hidden * 2u];
                r_c += hv * R[r_base + hidden * hidden * 3u];
                h_idx++;
            }}
            ct_prev = C_init[d_mul_batch_hidden + b_mul_hidden + h];
        }} else {{
            {y_prev_idx_calc}
            for (uint k = 0u; k < hidden; k++) {{
                float hv    = Y_prev[y_prev_idx + k];
                uint r_base = d_mul_h4_hidden + k * hidden + h;
                r_i += hv * R[r_base];
                r_o += hv * R[r_base + hidden * hidden];
                r_f += hv * R[r_base + hidden * hidden * 2u];
                r_c += hv * R[r_base + hidden * hidden * 3u];
            }}
            ct_prev = C_prev[y_prev_idx + h];
        }}

        {bias_logic}

        float it       = xw_i + r_i + b_wi + b_ri;
        float ft       = xw_f + r_f + b_wf + b_rf;
        float ct_tilde = xw_c + r_c + b_wc + b_rc;
        float ot       = xw_o + r_o + b_wo + b_ro;

        {peephole_i}
        {peephole_f}
        {clip_logic}

        if (d == 0u) {{
            it       = activation_f_0(it,       f_alpha_0, f_beta_0);
            ft       = activation_f_0(ft,       f_alpha_0, f_beta_0);
            ct_tilde = activation_g_0(ct_tilde, g_alpha_0, g_beta_0);
        }} else {{
            it       = activation_f_1(it,       f_alpha_1, f_beta_1);
            ft       = activation_f_1(ft,       f_alpha_1, f_beta_1);
            ct_tilde = activation_g_1(ct_tilde, g_alpha_1, g_beta_1);
        }}
        ct = ft * ct_prev + it * ct_tilde;

        {peephole_o}
        if (d == 0u) {{
            ot = activation_f_0(ot, f_alpha_0, f_beta_0);
            ht = ot * activation_h_0(ct, h_alpha_0, h_beta_0);
        }} else {{
            ot = activation_f_1(ot, f_alpha_1, f_beta_1);
            ht = ot * activation_h_1(ct, h_alpha_1, h_beta_1);
        }}
    }}

    {y_idx_calc}
    Y_curr[y_idx] = ht;
    C_curr[y_idx] = ct;

    if (seq_len_val > 0u) {{
        bool is_last = (!is_reverse && t == seq_len_val - 1u) || (is_reverse && t == 0u);
        if (is_last) {{
            Yh[d * batch * hidden + b * hidden + h] = ht;
            C_h[d * batch * hidden + b * hidden + h] = ct;
        }}
    }} else {{
        if (step_index == 0u) {{
            Yh[d * batch * hidden + b * hidden + h] = H_init[d * batch * hidden + b * hidden + h];
            C_h[d * batch * hidden + b * hidden + h] = C_init[d * batch * hidden + b * hidden + h];
        }}
    }}
}}
"""
        return compile_source(shader_source)

    def __repr__(self):
        return f"LstmOp({self.manager.get_device_properties()['device_name']}, dir={self.direction})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp is None:
                input_tensors.append((None, None))
                continue
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        tensor_Y, shape_Y = output_tensor_and_shape[0]
        tensor_Y_h = output_tensor_and_shape[1][0] if len(output_tensor_and_shape) > 1 else None
        tensor_Y_c = output_tensor_and_shape[2][0] if len(output_tensor_and_shape) > 2 else None

        seq = self.manager.sequence()
        valid_inputs = [t[0] for t in input_tensors if t[0] is not None]
        seq.record(kp.OpTensorSyncDevice(valid_inputs))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))

        out_tensors = [tensor_Y]
        if tensor_Y_h:
            out_tensors.append(tensor_Y_h)
        if tensor_Y_c:
            out_tensors.append(tensor_Y_c)
        seq.record(kp.OpTensorSyncLocal(out_tensors))
        seq.eval()

        Y_out   = tensor_Y.data().reshape(shape_Y)
        Y_h_out = tensor_Y_h.data().reshape(output_tensor_and_shape[1][1])
        Y_c_out = tensor_Y_c.data().reshape(output_tensor_and_shape[2][1])
        return Y_out, Y_h_out, Y_c_out

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        def make_param_uint(arr):
            """创建 uint32 参数 SSBO 并立即同步到 GPU。"""
            t = self.manager.tensor_t(np.array(arr, dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t])).eval()
            updated_tensors.append(t)
            return t

        def make_param_mixed(uint_vals, float_vals):
            """创建包含 uint32 + float32 混合的参数 SSBO（std430 布局按声明顺序排列）。"""
            data = np.array(uint_vals, dtype=np.uint32).tobytes() + \
                   np.array(float_vals, dtype=np.float32).tobytes()
            arr = np.frombuffer(data, dtype=np.uint32)
            t = self.manager.tensor_t(arr, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t])).eval()
            updated_tensors.append(t)
            return t

        t_X, s_X = input_tensors[0]
        t_W, s_W = input_tensors[1]
        t_R, s_R = input_tensors[2]

        t_B, s_B = (None, None)
        if len(input_tensors) > 3 and input_tensors[3][0] is not None:
            t_B, s_B = input_tensors[3]

        t_seq_lens, s_seq_lens = (None, None)
        if len(input_tensors) > 4 and input_tensors[4][0] is not None:
            t_seq_lens, s_seq_lens = input_tensors[4]

        t_init_h, s_init_h = (None, None)
        if len(input_tensors) > 5 and input_tensors[5][0] is not None:
            t_init_h, s_init_h = input_tensors[5]

        t_init_c, s_init_c = (None, None)
        if len(input_tensors) > 6 and input_tensors[6][0] is not None:
            t_init_c, s_init_c = input_tensors[6]

        t_P, s_P = (None, None)
        if len(input_tensors) > 7 and input_tensors[7][0] is not None:
            t_P, s_P = input_tensors[7]

        if self.layout == 0:
            seq_len    = s_X[0]
            batch_size = s_X[1]
            input_size = s_X[2]
        else:
            seq_len    = s_X[1]
            batch_size = s_X[0]
            input_size = s_X[2]

        num_directions = s_W[0]
        hidden_size_4  = s_W[1]
        hidden_size    = hidden_size_4 // 4

        if self.layout == 0:
            y_shape = [seq_len, num_directions, batch_size, hidden_size]
        else:
            y_shape = [batch_size, seq_len, num_directions, hidden_size]

        y_size = int(np.prod(y_shape))
        t_Y = self.manager.tensor(np.zeros(y_size, dtype=np.float32))
        updated_tensors.append(t_Y)

        t_C = self.manager.tensor(np.zeros(y_size, dtype=np.float32))
        updated_tensors.append(t_C)

        yh_shape = [num_directions, batch_size, hidden_size]
        ch_shape = [num_directions, batch_size, hidden_size]
        t_Y_h = self.manager.tensor(np.zeros(int(np.prod(yh_shape)), dtype=np.float32))
        updated_tensors.append(t_Y_h)
        t_C_h = self.manager.tensor(np.zeros(int(np.prod(ch_shape)), dtype=np.float32))
        updated_tensors.append(t_C_h)

        xw_size = seq_len * batch_size * num_directions * hidden_size_4
        t_XW = self.manager.tensor(np.zeros(xw_size, dtype=np.float32))
        updated_tensors.append(t_XW)

        # 初始隐藏状态
        if t_init_h is None:
            t_zero_h = self.manager.tensor(np.zeros(num_directions * batch_size * hidden_size, dtype=np.float32))
            updated_tensors.append(t_zero_h)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_zero_h])).eval()
            h_tensor = t_zero_h
        else:
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_init_h])).eval()
            h_tensor = t_init_h

        # 初始 cell state
        if t_init_c is None:
            t_zero_c = self.manager.tensor(np.zeros(num_directions * batch_size * hidden_size, dtype=np.float32))
            updated_tensors.append(t_zero_c)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_zero_c])).eval()
            c_tensor = t_zero_c
        else:
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_init_c])).eval()
            c_tensor = t_init_c

        # 虚拟偏置
        has_bias_bool = (t_B is not None)
        if t_B is None:
            t_B = self.manager.tensor(np.zeros(1, dtype=np.float32))
            updated_tensors.append(t_B)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_B])).eval()

        # 虚拟 peepholes
        has_peepholes_bool = (t_P is not None)
        if t_P is None:
            t_P = self.manager.tensor(np.zeros(1, dtype=np.float32))
            updated_tensors.append(t_P)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_P])).eval()

        # sequence_lens：uint32 tensor
        has_seq_lens_bool = (t_seq_lens is not None)
        if t_seq_lens is None:
            t_seq_lens = self.manager.tensor_t(np.zeros(1, dtype=np.uint32), kp.TensorTypes.device)
            updated_tensors.append(t_seq_lens)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_seq_lens])).eval()
        else:
            seq_lens_uint = t_seq_lens.data().astype(np.uint32)
            t_seq_lens = self.manager.tensor_t(seq_lens_uint, kp.TensorTypes.device)
            updated_tensors.append(t_seq_lens)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_seq_lens])).eval()

        # 1. 计算 XW = X @ W^T
        matmul_shader = self.matmul_shader_layout0 if self.layout == 0 else self.matmul_shader_layout1
        workgroup_matmul = (
            (batch_size    + LOCAL_X_3D - 1) // LOCAL_X_3D,
            (hidden_size_4 + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
            (seq_len       + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
        )
        for d in range(num_directions):
            param_matmul = make_param_uint([seq_len, batch_size, input_size, hidden_size_4, num_directions, d])
            updated_algorithms.append(self.manager.algorithm(
                [t_X, t_W, t_XW, param_matmul],
                matmul_shader,
                workgroup_matmul,
            ))

        # 2. 逐步运行 LSTM
        shader_key = (self.direction, self.layout, has_bias_bool, has_seq_lens_bool, has_peepholes_bool)
        step_shader = self._step_shaders[shader_key]

        clip_val = float(self.clip) if self.clip else 1e10

        if self.direction == "bidirectional":
            workgroup_step = (
                (batch_size  + LOCAL_X_3D - 1) // LOCAL_X_3D,
                (hidden_size + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                1,
            )
        else:
            workgroup_step = (
                (batch_size  + LOCAL_X_2D - 1) // LOCAL_X_2D,
                (hidden_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                1,
            )

        for step in range(seq_len):
            # Params buffer: uint batch, uint hidden, uint seq_len, uint step_index, float clip_val, float input_forget_f
            param_step = make_param_mixed(
                [batch_size, hidden_size, seq_len, step],
                [clip_val, float(self.input_forget)]
            )
            updated_algorithms.append(self.manager.algorithm(
                [t_XW, h_tensor, c_tensor, t_R, t_B, t_P,
                 t_Y, t_Y, t_C, t_C, t_Y_h, t_C_h, t_seq_lens, param_step],
                step_shader,
                workgroup_step,
            ))

        return [(t_Y, y_shape), (t_Y_h, yh_shape), (t_C_h, ch_shape)]
