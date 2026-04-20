import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class RnnOp:
    def __init__(self, manager: kp.Manager, direction="forward", hidden_size=None, activations=None, layout=0, activation_alpha=None, activation_beta=None):
        self.manager = manager
        self._step_shaders = {}
        self._shader_sources = {}

        # 用于 X * W^T 预计算的 matmul shader（layout 0：X 形状 [seq,batch,input]）
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
    uint hidden_size;
    uint num_directions;
    uint direction_index;
}};

void main() {{
    uint b = gl_GlobalInvocationID.x; // batch
    uint h = gl_GlobalInvocationID.y; // hidden
    uint s = gl_GlobalInvocationID.z; // seq

    if (b >= batch_size || h >= hidden_size || s >= seq_len) return;

    uint d = direction_index;
    float sum = 0.0;
    uint sb = s * batch_size + b;
    uint a_idx = sb * input_size;
    uint b_idx = (d * hidden_size + h) * input_size;

    for (uint k = 0u; k < input_size; k++, a_idx++, b_idx++) {{
        sum += A[a_idx] * B[b_idx];
    }}

    uint c_idx = (sb * num_directions + d) * hidden_size + h;
    C[c_idx] = sum;
}}
""")

        # 用于 X * W^T 预计算的 matmul shader（layout 1：X 形状 [batch,seq,input]）
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
    uint hidden_size;
    uint num_directions;
    uint direction_index;
}};

void main() {{
    uint b = gl_GlobalInvocationID.x; // batch
    uint h = gl_GlobalInvocationID.y; // hidden
    uint s = gl_GlobalInvocationID.z; // seq

    if (b >= batch_size || h >= hidden_size || s >= seq_len) return;

    uint d = direction_index;
    float sum = 0.0;
    uint a_idx = (b * seq_len + s) * input_size;
    uint b_idx = (d * hidden_size + h) * input_size;

    for (uint k = 0u; k < input_size; k++, a_idx++, b_idx++) {{
        sum += A[a_idx] * B[b_idx];
    }}

    uint sb = s * batch_size + b;
    uint c_idx = (sb * num_directions + d) * hidden_size + h;
    C[c_idx] = sum;
}}
""")

        self.set_attributes(
            direction=direction,
            hidden_size=hidden_size,
            activations=activations,
            layout=layout,
            activation_alpha=activation_alpha,
            activation_beta=activation_beta
        )

    def set_attributes(self, direction=None, layout=None, hidden_size=None, activations=None,
                       activation_alpha=None, activation_beta=None):
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
            self.activations = ["Tanh", "Tanh"]
        if activation_alpha is not None:
            self.activation_alpha = activation_alpha
        else:
            self.activation_alpha = []
        if activation_beta is not None:
            self.activation_beta = activation_beta
        else:
            self.activation_beta = []

        self._setup_activation_params()
        self._step_shaders = {}
        self._generate_all_step_shaders()

    def _generate_all_step_shaders(self):
        if self.direction == "bidirectional":
            for has_bias in [True, False]:
                for has_seq_lens in [True, False]:
                    key = (self.direction, self.layout, has_bias, has_seq_lens)
                    self._step_shaders[key] = self._generate_bidirectional_shader(
                        self.activations, self.layout, has_bias, has_seq_lens
                    )
        else:
            for has_bias in [True, False]:
                for has_seq_lens in [True, False]:
                    key = (self.direction, self.layout, has_bias, has_seq_lens)
                    self._step_shaders[key] = self._generate_unidirectional_shader(
                        self.direction, self.activations[0], self.layout, has_bias, has_seq_lens
                    )

    def _setup_activation_params(self):
        num_directions = 2 if self.direction == "bidirectional" else 1
        self.act_0_alpha = self.activation_alpha[0] if 0 < len(self.activation_alpha) else 0.0
        self.act_0_beta  = self.activation_beta[0]  if 0 < len(self.activation_beta)  else 0.0
        if num_directions == 2:
            self.act_1_alpha = self.activation_alpha[1] if 1 < len(self.activation_alpha) else 0.0
            self.act_1_beta  = self.activation_beta[1]  if 1 < len(self.activation_beta)  else 0.0

    def _generate_unidirectional_shader(self, direction, activation_name, layout_mode, has_bias, has_seq_lens):
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
        act_impl = activation_impls.get(activation_name.lower(), 'return tanh(val);')

        if direction == "forward":
            t_calc      = "uint t      = step_index;"
            t_prev_calc = "uint t_prev = t - 1u;"
        else:
            t_calc      = "uint t      = seq_len - 1u - step_index;"
            t_prev_calc = "uint t_prev = t + 1u;"

        if layout_mode == 0:
            y_idx_calc      = "uint y_idx      = t * batch * hidden + b * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = t_prev * batch * hidden + b * hidden;"
            xw_idx_calc     = "uint xw_idx     = t * batch * hidden + b * hidden + h;"
        else:
            y_idx_calc      = "uint y_idx      = b * seq_len * hidden + t * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = b * seq_len * hidden + t_prev * hidden;"
            xw_idx_calc     = "uint xw_idx     = t * batch * hidden + b * hidden + h;"

        seq_len_val_decl = "uint seq_len_val = uint(SeqLens[b]);" if has_seq_lens else "uint seq_len_val = seq_len;"
        bias_logic = """
    uint b_offset = 0u;
    val += Bias[b_offset + h] + Bias[b_offset + h + hidden];""" if has_bias else ""

        if direction == "forward":
            last_step_cond = "t == seq_len_val - 1u"
        else:
            last_step_cond = "t == 0u"

        shader_source = f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InXW     {{ float XW[];     }};
layout(std430, set = 0, binding = 1) readonly  buffer InHInit  {{ float H_init[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer InR      {{ float R[];      }};
layout(std430, set = 0, binding = 3) readonly  buffer InBias   {{ float Bias[];   }};
layout(std430, set = 0, binding = 4) writeonly buffer OutY     {{ float Y_curr[]; }};
layout(std430, set = 0, binding = 5) readonly  buffer InY      {{ float Y_prev[]; }};
layout(std430, set = 0, binding = 6) writeonly buffer OutYh    {{ float Yh[];     }};
layout(std430, set = 0, binding = 7) readonly  buffer InSeqLens{{ uint  SeqLens[];}};
layout(std430, set = 0, binding = 8) readonly  buffer Params   {{
    uint batch;
    uint hidden;
    uint seq_len;
    uint step_index;
}};

float apply_activation(float val, float alpha, float beta) {{
    {act_impl}
}}

void main() {{
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;

    if (b >= batch || h >= hidden) return;

    float alpha_val_f = {self.act_0_alpha};
    float beta_val_f  = {self.act_0_beta};

    {t_calc}
    {t_prev_calc}
    {seq_len_val_decl}

    bool is_padding = (t >= seq_len_val);
    float val = 0.0;

    if (!is_padding) {{
        {xw_idx_calc}
        val = XW[xw_idx];

        uint h_mul_hidden = h * hidden;
        uint b_mul_hidden = b * hidden;
        float r_sum = 0.0;
        if (step_index == 0u) {{
            uint h_idx = b_mul_hidden;
            uint r_idx = h_mul_hidden;
            for (uint k = 0u; k < hidden; k++) {{
                r_sum += H_init[h_idx++] * R[r_idx++];
            }}
        }} else {{
            {y_prev_idx_calc}
            uint y_ptr = y_prev_idx;
            uint r_idx = h_mul_hidden;
            for (uint k = 0u; k < hidden; k++) {{
                r_sum += Y_prev[y_ptr++] * R[r_idx++];
            }}
        }}
        val += r_sum;
        {bias_logic}
        val = apply_activation(val, alpha_val_f, beta_val_f);
    }}

    {y_idx_calc}
    Y_curr[y_idx] = val;

    if (seq_len_val > 0u) {{
        if ({last_step_cond}) {{
            Yh[b * hidden + h] = val;
        }}
    }} else {{
        if (step_index == 0u) {{
            Yh[b * hidden + h] = H_init[b * hidden + h];
        }}
    }}
}}
"""
        key = (direction, layout_mode, has_bias, has_seq_lens)
        self._shader_sources[key] = shader_source
        return compile_source(shader_source)

    def _generate_bidirectional_shader(self, act_names, layout_mode, has_bias, has_seq_lens):
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

        act_names_expanded = list(act_names[:2])
        while len(act_names_expanded) < 2:
            act_names_expanded.append("Tanh")

        act_functions = []
        for i, act_name in enumerate(act_names_expanded):
            impl = activation_impls.get(act_name.lower(), 'return tanh(val);')
            act_functions.append(f"""
float apply_activation_{i}(float val, float alpha, float beta) {{
    {impl}
}}""")
        activation_code = '\n'.join(act_functions)

        if layout_mode == 0:
            y_idx_calc      = "uint y_idx      = t * 2u * batch * hidden + d * batch * hidden + b * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = t_prev * 2u * batch * hidden + d * batch * hidden + b * hidden;"
            xw_idx_calc     = "uint xw_idx     = ((t * batch + b) * 2u + d) * hidden + h;"
        else:
            y_idx_calc      = "uint y_idx      = b * seq_len * 2u * hidden + t * 2u * hidden + d * hidden + h;"
            y_prev_idx_calc = "uint y_prev_idx = b * seq_len * 2u * hidden + t_prev * 2u * hidden + d * hidden;"
            xw_idx_calc     = "uint xw_idx     = ((t * batch + b) * 2u + d) * hidden + h;"

        seq_len_val_decl = "uint seq_len_val = SeqLens[b];" if has_seq_lens else "uint seq_len_val = seq_len;"
        bias_logic = """
        uint b_offset = d * 2u * hidden;
        val += Bias[b_offset + h] + Bias[b_offset + h + hidden];""" if has_bias else ""

        shader_source = f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InXW     {{ float XW[];     }};
layout(std430, set = 0, binding = 1) readonly  buffer InHInit  {{ float H_init[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer InR      {{ float R[];      }};
layout(std430, set = 0, binding = 3) readonly  buffer InBias   {{ float Bias[];   }};
layout(std430, set = 0, binding = 4) writeonly buffer OutY     {{ float Y_curr[]; }};
layout(std430, set = 0, binding = 5) readonly  buffer InY      {{ float Y_prev[]; }};
layout(std430, set = 0, binding = 6) writeonly buffer OutYh    {{ float Yh[];     }};
layout(std430, set = 0, binding = 7) readonly  buffer InSeqLens{{ uint  SeqLens[];}};
layout(std430, set = 0, binding = 8) readonly  buffer Params   {{
    uint batch;
    uint hidden;
    uint seq_len;
    uint step_index;
}};

{activation_code}

void main() {{
    uint b = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    uint d = gl_GlobalInvocationID.z;  // 方向索引：0=前向，1=反向

    if (b >= batch || h >= hidden || d >= 2u) return;

    float alpha_0 = {self.act_0_alpha};
    float beta_0  = {self.act_0_beta};
    float alpha_1 = {self.act_1_alpha};
    float beta_1  = {self.act_1_beta};

    bool is_reverse = (d == 1u);
    uint t      = is_reverse ? (seq_len - 1u - step_index) : step_index;
    uint t_prev = is_reverse ? (t + 1u) : (t - 1u);

    {seq_len_val_decl}

    bool is_padding = (t >= seq_len_val);
    float val = 0.0;

    if (!is_padding) {{
        {xw_idx_calc}
        val = XW[xw_idx];

        uint d_mul_batch_hidden  = d * batch * hidden;
        uint b_mul_hidden        = b * hidden;
        uint d_mul_hidden_hidden = d * hidden * hidden;
        uint h_mul_hidden        = h * hidden;

        float r_sum = 0.0;
        if (step_index == 0u) {{
            uint h_idx = d_mul_batch_hidden + b_mul_hidden;
            uint r_idx = d_mul_hidden_hidden + h_mul_hidden;
            for (uint k = 0u; k < hidden; k++) {{
                r_sum += H_init[h_idx++] * R[r_idx++];
            }}
        }} else {{
            {y_prev_idx_calc}
            uint y_ptr = y_prev_idx;
            uint r_idx = d_mul_hidden_hidden + h_mul_hidden;
            for (uint k = 0u; k < hidden; k++) {{
                r_sum += Y_prev[y_ptr++] * R[r_idx++];
            }}
        }}
        val += r_sum;
        {bias_logic}
        float alpha = (d == 0u) ? alpha_0 : alpha_1;
        float beta  = (d == 0u) ? beta_0  : beta_1;
        val = (d == 0u) ? apply_activation_0(val, alpha, beta)
                        : apply_activation_1(val, alpha, beta);
    }}

    {y_idx_calc}
    Y_curr[y_idx] = val;

    if (seq_len_val > 0u) {{
        bool is_last = (!is_reverse && t == seq_len_val - 1u) || (is_reverse && t == 0u);
        if (is_last) {{
            Yh[d * batch * hidden + b * hidden + h] = val;
        }}
    }} else {{
        if (step_index == 0u) {{
            Yh[d * batch * hidden + b * hidden + h] = H_init[d * batch * hidden + b * hidden + h];
        }}
    }}
}}
"""
        key = (self.direction, layout_mode, has_bias, has_seq_lens)
        self._shader_sources[key] = shader_source
        return compile_source(shader_source)

    def __repr__(self):
        return f"RnnOp({self.manager.get_device_properties()['device_name']}, dir={self.direction})"

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

        seq = self.manager.sequence()
        valid_inputs = [t[0] for t in input_tensors if t[0] is not None]
        seq.record(kp.OpTensorSyncDevice(valid_inputs))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))

        out_tensors = [tensor_Y]
        if tensor_Y_h:
            out_tensors.append(tensor_Y_h)
        seq.record(kp.OpTensorSyncLocal(out_tensors))
        seq.eval()

        Y_out   = tensor_Y.data().reshape(shape_Y)
        Y_h_out = tensor_Y_h.data().reshape(output_tensor_and_shape[1][1])
        return Y_out, Y_h_out

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        def make_param(arr):
            """创建 uint32 参数 SSBO 并立即同步到 GPU。"""
            t = self.manager.tensor_t(np.array(arr, dtype=np.uint32), kp.TensorTypes.device)
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

        if self.layout == 0:
            seq_len    = s_X[0]
            batch_size = s_X[1]
            input_size = s_X[2]
        else:
            seq_len    = s_X[1]
            batch_size = s_X[0]
            input_size = s_X[2]

        num_directions = s_W[0]
        hidden_size    = s_W[1]

        if self.layout == 0:
            y_shape = [seq_len, num_directions, batch_size, hidden_size]
        else:
            y_shape = [batch_size, seq_len, num_directions, hidden_size]

        y_size = int(np.prod(y_shape))
        t_Y = self.manager.tensor(np.zeros(y_size, dtype=np.float32))
        updated_tensors.append(t_Y)

        yh_shape = [num_directions, batch_size, hidden_size]
        t_Y_h = self.manager.tensor(np.zeros(int(np.prod(yh_shape)), dtype=np.float32))
        updated_tensors.append(t_Y_h)

        xw_size = seq_len * batch_size * num_directions * hidden_size
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

        # 虚拟偏置
        has_bias_bool = (t_B is not None)
        if t_B is None:
            t_B = self.manager.tensor(np.zeros(1, dtype=np.float32))
            updated_tensors.append(t_B)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_B])).eval()

        # sequence_lens：用 uint32 tensor
        has_seq_lens_bool = (t_seq_lens is not None)
        if t_seq_lens is None:
            t_seq_lens = self.manager.tensor_t(np.zeros(1, dtype=np.uint32), kp.TensorTypes.device)
            updated_tensors.append(t_seq_lens)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_seq_lens])).eval()
        else:
            # 将 float32 的 seq_lens 转成 uint32 tensor 供 shader 使用
            seq_lens_uint = t_seq_lens.data().astype(np.uint32)
            t_seq_lens = self.manager.tensor_t(seq_lens_uint, kp.TensorTypes.device)
            updated_tensors.append(t_seq_lens)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t_seq_lens])).eval()

        # 1. 计算 XW = X @ W^T（每个方向单独调度）
        matmul_shader = self.matmul_shader_layout0 if self.layout == 0 else self.matmul_shader_layout1
        workgroup_matmul = (
            (batch_size  + LOCAL_X_3D - 1) // LOCAL_X_3D,
            (hidden_size + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
            (seq_len     + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
        )
        for d in range(num_directions):
            param_matmul = make_param([seq_len, batch_size, input_size, hidden_size, num_directions, d])
            updated_algorithms.append(self.manager.algorithm(
                [t_X, t_W, t_XW, param_matmul],
                matmul_shader,
                workgroup_matmul,
            ))

        # 2. 逐步运行 RNN
        shader_key = (self.direction, self.layout, has_bias_bool, has_seq_lens_bool)
        step_shader = self._step_shaders[shader_key]

        if self.direction == "bidirectional":
            workgroup_step = (
                (batch_size  + LOCAL_X_3D - 1) // LOCAL_X_3D,
                (hidden_size + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                1,  # z 轴：两个方向（0 和 1），local_size_z=LOCAL_Z_3D 已足够覆盖 2
            )
        else:
            workgroup_step = (
                (batch_size  + LOCAL_X_2D - 1) // LOCAL_X_2D,
                (hidden_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                1,
            )

        for step in range(seq_len):
            param_step = make_param([batch_size, hidden_size, seq_len, step])
            updated_algorithms.append(self.manager.algorithm(
                [t_XW, h_tensor, t_R, t_B, t_Y, t_Y, t_Y_h, t_seq_lens, param_step],
                step_shader,
                workgroup_step,
            ))

        return [(t_Y, y_shape), (t_Y_h, yh_shape)]
