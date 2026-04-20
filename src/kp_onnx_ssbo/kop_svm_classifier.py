import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D


class SVMClassifierOp:

    def __init__(self,
                 manager: kp.Manager,
                 classlabels_ints=None,
                 classlabels_strings=None,
                 coefficients=None,
                 kernel_params=None,
                 kernel_type="LINEAR",
                 post_transform="NONE",
                 prob_a=None,
                 prob_b=None,
                 rho=None,
                 support_vectors=None,
                 vectors_per_class=None):
        self.classlabels_ints = classlabels_ints
        self.classlabels_strings = classlabels_strings
        self.coefficients = coefficients
        self.kernel_params = kernel_params
        self.kernel_type = kernel_type
        self.post_transform = post_transform
        self.prob_a = prob_a
        self.prob_b = prob_b
        self.rho = rho
        self.support_vectors = support_vectors
        self.vectors_per_class = vectors_per_class
        self.manager = manager

        # Shader for Linear Kernel (no support vectors)
        self.linear_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf  {{ float in_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer CoefBuf {{ float coef_buf[]; }};
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_buf[]; }};
layout(std430, set = 0, binding = 3) readonly buffer Params {{
    uint feature_size;
    uint class_count;
    uint num_samples;
    float rho0;
}};

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples * class_count) return;
    
    uint sample_idx = idx / class_count;
    uint class_idx = idx - sample_idx * class_count;
    
    uint in_idx = sample_idx * feature_size;
    uint coef_idx = class_idx * feature_size;
    
    float s = 0.0;
    // float c = 0.0; // Kahan summation removed for simplicity unless required precision is high
    for (uint j = 0; j < feature_size; ++j) {{
        s += in_buf[in_idx] * coef_buf[coef_idx];
        ++in_idx;
        ++coef_idx;
    }}
    
    out_buf[idx] = s + rho0;
}}
""")

        # Common preamble for SVC shaders
        # Bindings:
        # 0: InBuf
        # 1: SVBuf
        # 2: CoefBuf
        # 3: RhoBuf
        # 4: StartVecBuf
        # 5: VecPerClassBuf
        # 6: ScoreBuf
        # 7: Params
        svc_layout = f"""
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf {{ float in_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer SVBuf {{ float sv_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer CoefBuf {{ float coef_buf[]; }};
layout(std430, set = 0, binding = 3) readonly buffer RhoBuf {{ float rho_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer StartVecBuf {{ float start_vec_buf[]; }};
layout(std430, set = 0, binding = 5) readonly buffer VecPerClassBuf {{ float vec_per_class_buf[]; }};
layout(std430, set = 0, binding = 6) writeonly buffer ScoreBuf {{ float score_buf[]; }};
layout(std430, set = 0, binding = 7) readonly buffer Params {{
    uint feature_size;
    uint class_count;
    uint vector_count;
    uint num_pairs;
    uint num_samples;
    float gamma;
    float coef0;
    float degree;
}};
"""

        svc_main_pre = """
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples * num_pairs) return;

    uint sample_idx = idx / num_pairs;
    uint pair_idx = idx - sample_idx * num_pairs;

    uint i = 0;
    uint j = 1;
    uint count = 0;
    for (uint ci = 0; ci < class_count - 1; ++ci) {
        uint pairs_for_ci = class_count - ci - 1;
        if (count + pairs_for_ci > pair_idx) {
            i = ci;
            j = ci + 1 + (pair_idx - count);
            break;
        }
        count += pairs_for_ci;
    }
    
    uint in_offset = sample_idx * feature_size;
    uint si_i = uint(start_vec_buf[i]);
    uint si_j = uint(start_vec_buf[j]);
    uint class_i_sc = uint(vec_per_class_buf[i]);
    uint class_j_sc = uint(vec_per_class_buf[j]);
    uint coef_base_i = (j - 1) * vector_count;
    uint coef_base_j = i * vector_count;
"""

        self.svc_linear_shader = compile_source(f"""
#version 450
{svc_layout}
{svc_main_pre}
    
    float s1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {{
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_offset + k] * sv_buf[sv_offset + k];
        }}
        
        s1 += coef_buf[coef_base_i + sv_idx] * dot_val;
    }}
    
    float s2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {{
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_offset + k] * sv_buf[sv_offset + k];
        }}
        
        s2 += coef_buf[coef_base_j + sv_idx] * dot_val;
    }}
    
    score_buf[idx] = rho_buf[pair_idx] + s1 + s2;
}}
""")
        self.poly_shader = compile_source(f"""
#version 450
{svc_layout}
{svc_main_pre}

    uint degree_int = uint(degree);

    float s1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {{
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_offset + k] * sv_buf[sv_offset + k];
        }}
        
        float kernel_val = dot_val * gamma + coef0;
        float powered = 1.0;
        for (uint d = 0; d < degree_int; ++d) {{
            powered *= kernel_val;
        }}
        
        s1 += coef_buf[coef_base_i + sv_idx] * powered;
    }}
    
    float s2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {{
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_offset + k] * sv_buf[sv_offset + k];
        }}
        
        float kernel_val = dot_val * gamma + coef0;
        float powered = 1.0;
        for (uint d = 0; d < degree_int; ++d) {{
            powered *= kernel_val;
        }}
        
        s2 += coef_buf[coef_base_j + sv_idx] * powered;
    }}
    
    score_buf[idx] = rho_buf[pair_idx] + s1 + s2;
}}
""")
        self.rbf_shader = compile_source(f"""
#version 450
{svc_layout}
{svc_main_pre}

    float neg_gamma = -gamma;

    float s1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {{
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float squared_dist = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            float diff = in_buf[in_offset + k] - sv_buf[sv_offset + k];
            squared_dist += diff * diff;
        }}
        
        s1 += coef_buf[coef_base_i + sv_idx] * exp(neg_gamma * squared_dist);
    }}
    
    float s2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {{
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float squared_dist = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            float diff = in_buf[in_offset + k] - sv_buf[sv_offset + k];
            squared_dist += diff * diff;
        }}
        
        s2 += coef_buf[coef_base_j + sv_idx] * exp(neg_gamma * squared_dist);
    }}
    
    score_buf[idx] = rho_buf[pair_idx] + s1 + s2;
}}
""")
        self.sigmoid_shader = compile_source(f"""
#version 450
{svc_layout}
{svc_main_pre}

    float s1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {{
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_offset + k] * sv_buf[sv_offset + k];
        }}
        
        s1 += coef_buf[coef_base_i + sv_idx] * tanh(dot_val * gamma + coef0);
    }}
    
    float s2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {{
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_offset + k] * sv_buf[sv_offset + k];
        }}
        
        s2 += coef_buf[coef_base_j + sv_idx] * tanh(dot_val * gamma + coef0);
    }}
    
    score_buf[idx] = rho_buf[pair_idx] + s1 + s2;
}}
""")
        self.vote_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly buffer ScoreBuf {{ float score_buf[]; }};
layout(std430, set = 0, binding = 1) writeonly buffer VoteBuf {{ float vote_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer Params {{
    uint num_samples;
    uint class_count;
    uint num_pairs;
}};

void main() {{
    uint sample_idx = gl_GlobalInvocationID.x;
    if (sample_idx >= num_samples) return;
    
    uint vote_base = sample_idx * class_count;
    
    // Initialize votes to 0 for this sample
    for (uint c = 0; c < class_count; ++c) {{
        vote_buf[vote_base + c] = 0.0;
    }}
    
    // Compute votes from scores
    uint score_base = sample_idx * num_pairs;
    uint pair_idx = 0;
    
    for (uint i = 0; i < class_count - 1; ++i) {{
        for (uint j = i + 1; j < class_count; ++j) {{
            if (score_buf[score_base + pair_idx] > 0.0) {{
                vote_buf[vote_base + i] += 1.0;
            }} else {{
                vote_buf[vote_base + j] += 1.0;
            }}
            ++pair_idx;
        }}
    }}
}}
""")
        self.softmax_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) buffer ScoreBuf {{ float score_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer Params {{
    uint num_samples;
    uint num_classes;
}};

void main() {{
    uint sample_idx = gl_GlobalInvocationID.x;
    if (sample_idx >= num_samples) return;
    
    uint base_idx = sample_idx * num_classes;
    
    // Find max value
    float v_max = score_buf[base_idx];
    for (uint i = 1; i < num_classes; ++i) {{
        float val = score_buf[base_idx + i];
        if (val > v_max) {{
            v_max = val;
        }}
    }}
    
    // Compute exp(val - v_max) and sum
    float sum_exp = 0.0;
    for (uint i = 0; i < num_classes; ++i) {{
        uint idx = base_idx + i;
        score_buf[idx] = exp(score_buf[idx] - v_max);
        sum_exp += score_buf[idx];
    }}
    
    // Normalize
    for (uint i = 0; i < num_classes; ++i) {{
        score_buf[base_idx + i] /= sum_exp;
    }}
}}
""")
        self.logistic_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) buffer ScoreBuf {{ float score_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer Params {{
    uint num_samples;
    uint num_classes;
}};

void main() {{
    uint sample_idx = gl_GlobalInvocationID.x;
    if (sample_idx >= num_samples) return;
    
    uint base_idx = sample_idx * num_classes;
    
    for (uint i = 0; i < num_classes; ++i) {{
        uint idx = base_idx + i;
        float val = score_buf[idx];
        float abs_val = abs(val);
        float v = 1.0 / (1.0 + exp(-abs_val));
        score_buf[idx] = (val < 0.0) ? (1.0 - v) : v;
    }}
}}
""")
        self.softmax_zero_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) buffer ScoreBuf {{ float score_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer Params {{
    uint num_samples;
    uint num_classes;
}};

void main() {{
    uint sample_idx = gl_GlobalInvocationID.x;
    if (sample_idx >= num_samples) return;
    
    uint base_idx = sample_idx * num_classes;
    
    // Find max value
    float v_max = score_buf[base_idx];
    for (uint i = 1; i < num_classes; ++i) {{
        float val = score_buf[base_idx + i];
        if (val > v_max) {{
            v_max = val;
        }}
    }}
    
    float exp_neg_v_max = exp(-v_max);
    float sum_val = 0.0;
    
    for (uint i = 0; i < num_classes; ++i) {{
        uint idx = base_idx + i;
        float v = score_buf[idx];
        if (v > 0.0000001 || v < -0.0000001) {{
            score_buf[idx] = exp(v - v_max);
        }} else {{
            score_buf[idx] = v * exp_neg_v_max;
        }}
        sum_val += score_buf[idx];
    }}
    
    // Normalize
    float norm_val = (sum_val == 0.0) ? 0.5 : (1.0 / sum_val);
    for (uint i = 0; i < num_classes; ++i) {{
        score_buf[base_idx + i] *= norm_val;
    }}
}}
""")
        self.probit_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) buffer ScoreBuf {{ float score_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer Params {{
    uint num_samples;
    uint num_classes;
}};

const float PI = 3.14159265358979323846;
const float SQRT2 = 1.41421356;

float erf_inv(float x) {{
    float sgn = (x < 0.0) ? -1.0 : 1.0;
    x = (1.0 - x) * (1.0 + x);
    if (x == 0.0) return 0.0;
    
    float log_val = log(x);
    float v = 2.0 / (PI * 0.147) + 0.5 * log_val;
    float v2 = (1.0 / 0.147) * log_val;
    float v3 = -v + sqrt(v * v - v2);
    return sgn * sqrt(v3);
}}

void main() {{
    uint sample_idx = gl_GlobalInvocationID.x;
    if (sample_idx >= num_samples) return;
    
    uint base_idx = sample_idx * num_classes;
    
    for (uint i = 0; i < num_classes; ++i) {{
        uint idx = base_idx + i;
        float val = score_buf[idx];
        score_buf[idx] = SQRT2 * erf_inv(val * 2.0 - 1.0);
    }}
}}
""")
        self.argmax_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf {{ float in_data[]; }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf {{ uint out_data[]; }};
layout(std430, set = 0, binding = 2) readonly buffer Params {{
    uint axis_size;
    uint block_size;
    uint bound_x; // mapped to block_size/batch_size (gx)
}};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    if (gx >= bound_x) return;
    
    // Logic: we have [bound_x, axis_size, block_size] tensor if using 3D,
    // but here we likely have [num_samples, class_count] or similar structure.
    // Original argmax was dispatch(n_samples, 1, 1).
    // Here we treat it as 1D dispatch over samples (bound_x = num_samples)
    // axis_size = class_count
    // block_size = 1 (effectively)
    
    // Re-check original usage:
    // argmax_shader dispatch(n_samples, 1, 1) params [class_count, 1.0] => axis_size=class_count, block_size=1
    // The original logic was:
    // uint gx = gl_GlobalInvocationID.x;
    // uint gy = gl_GlobalInvocationID.y; // 0
    // base_idx = gx * axis_size * block_size + gy
    
    uint gy = 0;
    uint base_idx = gx * axis_size * block_size + gy;
    
    float max_val = in_data[base_idx];
    uint max_idx = 0u;
    base_idx += block_size;
    
    for (uint i = 1u; i < axis_size; ++i, base_idx += block_size) {{
        if (in_data[base_idx] > max_val) {{
            max_val = in_data[base_idx];
            max_idx = i;
        }}
    }}
    out_data[gx * block_size + gy] = max_idx;
}}
""")
        self.sigmoid_prob_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) buffer ScoreBuf {{ float score_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer ProbABuf {{ float prob_a_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer ProbBBuf {{ float prob_b_buf[]; }};
layout(std430, set = 0, binding = 3) readonly buffer Params {{
    uint num_samples;
    uint num_pairs;
}};

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples * num_pairs) return;
    
    uint sample_idx = idx / num_pairs;
    uint pair_idx = idx - sample_idx * num_pairs;
    
    float score = score_buf[idx];
    float val = score * prob_a_buf[pair_idx] + prob_b_buf[pair_idx];
    float abs_val = abs(val);
    float logistic_val = 1.0 / (1.0 + exp(-abs_val));
    logistic_val = (val < 0.0) ? (1.0 - logistic_val) : logistic_val;
    score_buf[idx] = 1.0 - logistic_val;
}}
""")
        self.label_map_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly buffer IndicesBuf {{ uint indices_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer LabelsBuf {{ float labels_buf[]; }};
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_buf[]; }};
layout(std430, set = 0, binding = 3) readonly buffer Params {{
    uint bound_x;
}};

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= bound_x) return;
    uint label_idx = indices_buf[idx];
    out_buf[idx] = labels_buf[label_idx];
}}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"SVMClassifierOp({{dev}})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensors_and_shapes = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        if updated_algorithms:
            seq = self.manager.sequence()
            output_tensor_set = {id(ot) for ot, _ in output_tensors_and_shapes}

            # Sync all inputs and updated tensors
            tensors_to_sync = [t for t, _ in input_tensors]
            for t in updated_tensors:
                if id(t) not in output_tensor_set:
                    tensors_to_sync.append(t)

            seq.record(kp.OpTensorSyncDevice(tensors_to_sync))

            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))

            seq.record(kp.OpTensorSyncLocal([t for t, _ in output_tensors_and_shapes]))
            seq.eval()

        outputs = []
        for i, (tensor_out, output_shape) in enumerate(output_tensors_and_shapes):
            output = tensor_out.data().reshape(output_shape)
            if i == 0:
                output = output.astype(np.int64)
            outputs.append(output)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors

        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, in_shape = input_tensors[0]
        n_samples = in_shape[0]
        feature_size = int(np.prod(in_shape[1:]))

        gamma, coef0, degree = (
            (self.kernel_params[0], self.kernel_params[1], self.kernel_params[2])
            if self.kernel_params else (0.0, 0.0, 0)
        )

        classlabels = self.classlabels_ints if self.classlabels_ints is not None else self.classlabels_strings
        class_count = max(len(classlabels or []), 1)

        post_transform_upper = (self.post_transform or "NONE").upper()
        has_proba = self.prob_a is not None and self.prob_b is not None

        vector_count = 0
        starting_vector = []
        if self.vectors_per_class is not None:
            for vc in self.vectors_per_class:
                starting_vector.append(vector_count)
                vector_count += vc

        if vector_count > 0:
            # SVC mode
            kernel_upper = self.kernel_type.upper()

            sv_data = self.support_vectors.astype(np.float32).reshape(-1)
            coef_data = self.coefficients.astype(np.float32).reshape(-1)

            num_pairs = class_count * (class_count - 1) // 2

            sv_tensor = self.manager.tensor(sv_data)
            coef_tensor = self.manager.tensor(coef_data)
            rho_tensor = self.manager.tensor(self.rho)
            start_vec_tensor = self.manager.tensor(np.array(starting_vector, dtype=np.float32))
            vec_per_class_tensor = self.manager.tensor(np.array(self.vectors_per_class, dtype=np.float32))
            score_tensor = self.manager.tensor(np.zeros(n_samples * num_pairs, dtype=np.float32))

            updated_tensors.extend([
                sv_tensor, coef_tensor, rho_tensor, start_vec_tensor, vec_per_class_tensor, score_tensor
            ])

            # Prepare Params
            params = np.zeros(8, dtype=np.float32) # Stored as float but interpreted as mixture in shader (layout is consistent)
            # Layout: uint feature_size, class_count, vector_count, num_pairs, num_samples, float gamma, coef0, degree
            params_uint = params.view(np.uint32)
            params_uint[0] = feature_size
            params_uint[1] = class_count
            params_uint[2] = vector_count
            params_uint[3] = num_pairs
            params_uint[4] = n_samples
            # floats
            params[5] = gamma
            params[6] = coef0
            params[7] = float(degree)

            params_tensor = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_tensor])).eval()

            # Step 1: Compute scores
            total_threads = n_samples * num_pairs
            workgroup = ((total_threads + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            shader = None
            if kernel_upper == "LINEAR":
                shader = self.svc_linear_shader
            elif kernel_upper == "POLY":
                shader = self.poly_shader
            elif kernel_upper == "RBF":
                shader = self.rbf_shader
            elif kernel_upper == "SIGMOID":
                shader = self.sigmoid_shader
            else:
                raise ValueError(f"Unsupported kernel type: {kernel_upper}")

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, sv_tensor, coef_tensor, rho_tensor, start_vec_tensor, vec_per_class_tensor, score_tensor, params_tensor],
                shader,
                workgroup
            ))

            # Step 1.5: Apply probability transformation if has_proba
            if has_proba:
                prob_a_tensor = self.manager.tensor(self.prob_a)
                prob_b_tensor = self.manager.tensor(self.prob_b)
                updated_tensors.extend([prob_a_tensor, prob_b_tensor])

                params_prob = self.manager.tensor_t(np.array([n_samples, num_pairs], dtype=np.uint32), kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([params_prob])).eval()

                updated_algorithms.append(
                    self.manager.algorithm(
                        [score_tensor, prob_a_tensor, prob_b_tensor, params_prob],
                        self.sigmoid_prob_shader,
                        workgroup  # Same workgroup size as scores (n_samples * num_pairs)
                    )
                )

            # Step 2: Compute votes from scores (GPU implementation)
            vote_tensor = self.manager.tensor(np.zeros(n_samples * class_count, dtype=np.float32))
            updated_tensors.append(vote_tensor)

            params_vote = self.manager.tensor_t(np.array([n_samples, class_count, num_pairs], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_vote])).eval()

            workgroup_vote = ((n_samples + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            updated_algorithms.append(
                self.manager.algorithm(
                    [score_tensor, vote_tensor, params_vote],
                    self.vote_shader,
                    workgroup_vote
                )
            )

            # Step 3: Argmax on votes to get label indices
            label_indices_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.uint32))
            updated_tensors.append(label_indices_tensor)

            params_argmax = self.manager.tensor_t(np.array([class_count, 1, n_samples], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_argmax])).eval()

            workgroup_argmax = ((n_samples + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            updated_algorithms.append(
                self.manager.algorithm(
                    [vote_tensor, label_indices_tensor, params_argmax],
                    self.argmax_shader,
                    workgroup_argmax
                )
            )

            # Map indices to actual labels
            if self.classlabels_strings is not None or classlabels is None or len(classlabels) == 0:
                labels_lut_data = [i for i in range(class_count)]
            else:
                labels_lut_data = classlabels

            labels_lut_tensor = self.manager.tensor(np.array(labels_lut_data, dtype=np.float32))
            labels_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            updated_tensors.extend([labels_lut_tensor, labels_tensor])

            params_label = self.manager.tensor_t(np.array([n_samples], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_label])).eval()

            updated_algorithms.append(
                self.manager.algorithm(
                    [label_indices_tensor, labels_lut_tensor, labels_tensor, params_label],
                    self.label_map_shader,
                    workgroup_argmax
                )
            )

            # 设置用于post_transform的tensor和相关参数
            transform_tensor = vote_tensor
            output_score_shape = (n_samples, num_pairs)
            num_classes_for_post = class_count

        else:
            # SVM_LINEAR mode: vector_count == 0
            # Formula: for each class j, score[j] = X · coefficients[j] + rho[0]
            # Output: (n_samples, class_count) scores

            coef_data = self.coefficients.astype(np.float32).reshape(-1)
            rho0 = self.rho[0] if self.rho is not None and len(self.rho) > 0 else 0.0

            coef_tensor = self.manager.tensor(coef_data)
            score_tensor = self.manager.tensor(np.zeros(n_samples * class_count, dtype=np.float32))
            updated_tensors.append(coef_tensor)
            updated_tensors.append(score_tensor)

            # Params: feature_size, class_count, num_samples, rho0 (float)
            # 混合类型：前3个uint，最后1个float，打包成float32数组传输
            params_linear = np.zeros(4, dtype=np.float32)
            params_linear.view(np.uint32)[0] = feature_size
            params_linear.view(np.uint32)[1] = class_count
            params_linear.view(np.uint32)[2] = n_samples
            params_linear[3] = rho0

            params_linear_tensor = self.manager.tensor_t(params_linear, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_linear_tensor])).eval()

            total_elements = n_samples * class_count
            workgroup_linear = ((total_elements + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, coef_tensor, score_tensor, params_linear_tensor],
                self.linear_shader,
                workgroup_linear
            ))

            # In LINEAR mode, scores are (n_samples, class_count)
            # We use argmax directly on scores
            label_indices_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.uint32))
            updated_tensors.append(label_indices_tensor)

            params_argmax = self.manager.tensor_t(np.array([class_count, 1, n_samples], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_argmax])).eval()

            workgroup_argmax = ((n_samples + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            updated_algorithms.append(
                self.manager.algorithm(
                    [score_tensor, label_indices_tensor, params_argmax],
                    self.argmax_shader,
                    workgroup_argmax
                )
            )

            # Map indices to actual labels
            if self.classlabels_strings is not None or classlabels is None or len(classlabels) == 0:
                labels_lut_data = [i for i in range(class_count)]
            else:
                labels_lut_data = classlabels

            labels_lut_tensor = self.manager.tensor(np.array(labels_lut_data, dtype=np.float32))
            labels_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            updated_tensors.extend([labels_lut_tensor, labels_tensor])

            params_label = self.manager.tensor_t(np.array([n_samples], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_label])).eval()

            updated_algorithms.append(
                self.manager.algorithm(
                    [label_indices_tensor, labels_lut_tensor, labels_tensor, params_label],
                    self.label_map_shader,
                    workgroup_argmax
                )
            )

            # 设置用于post_transform的tensor和相关参数
            transform_tensor = score_tensor
            output_score_shape = (n_samples, class_count)
            num_classes_for_post = class_count

        # 应用 post_transform（两个分支共享）
        if post_transform_upper != "NONE":
            params_post = self.manager.tensor_t(np.array([n_samples, num_classes_for_post], dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_post])).eval()

            workgroup_post = ((n_samples + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            shader_post = None
            if post_transform_upper == "SOFTMAX":
                shader_post = self.softmax_shader
            elif post_transform_upper == "LOGISTIC":
                shader_post = self.logistic_shader
            elif post_transform_upper == "SOFTMAX_ZERO":
                shader_post = self.softmax_zero_shader
            elif post_transform_upper == "PROBIT":
                shader_post = self.probit_shader

            if shader_post:
                updated_algorithms.append(
                    self.manager.algorithm(
                        [transform_tensor, params_post],
                        shader_post,
                        workgroup_post
                    )
                )

        return [
            (labels_tensor, [n_samples]),
            (score_tensor, list(output_score_shape))
        ]

