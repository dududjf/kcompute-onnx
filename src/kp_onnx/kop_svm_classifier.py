import kp
import numpy as np
from .shader_utils import compile_source


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
        self.linear_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 2) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float class_count_f = 0.0;
layout(constant_id = 3) const float rho0_f = 0.0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint feature_size = uint(feature_size_f);
    uint class_count = uint(class_count_f);
    
    uint sample_idx = idx / class_count;
    uint class_idx = idx - sample_idx * class_count;
    
    uint in_idx = sample_idx * feature_size;
    uint coef_idx = class_idx * feature_size;
    
    float s = 0.0;
    float c = 0.0;
    for (uint j = 0; j < feature_size; ++j) {
        float prod = in_buf[in_idx] * coef_buf[coef_idx];
        float y = prod - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
        ++in_idx;
        ++coef_idx;
    }
    
    out_buf[idx] = s + rho0_f;
}
""")
        self.svc_linear_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer SVBuf { float sv_buf[]; };
layout(set = 0, binding = 2) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 3) readonly buffer RhoBuf { float rho_buf[]; };
layout(set = 0, binding = 4) readonly buffer StartVecBuf { float start_vec_buf[]; };
layout(set = 0, binding = 5) readonly buffer VecPerClassBuf { float vec_per_class_buf[]; };
layout(set = 0, binding = 6) writeonly buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float class_count_f = 0.0;
layout(constant_id = 3) const float vector_count_f = 0.0;
layout(constant_id = 4) const float num_pairs_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint pair_idx = gl_GlobalInvocationID.y;
    uint feature_size = uint(feature_size_f);
    uint class_count = uint(class_count_f);
    uint vector_count = uint(vector_count_f);
    uint num_pairs = uint(num_pairs_f);

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
    
    float s1 = 0.0;
    float c1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        float dot_c = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float prod = in_buf[in_offset + k] * sv_buf[sv_offset + k];
            float y = prod - dot_c;
            float t = dot_val + y;
            dot_c = (t - dot_val) - y;
            dot_val = t;
        }
        
        float term = coef_buf[coef_base_i + sv_idx] * dot_val;
        float y = term - c1;
        float t = s1 + y;
        c1 = (t - s1) - y;
        s1 = t;
    }
    
    float s2 = 0.0;
    float c2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        float dot_c = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float prod = in_buf[in_offset + k] * sv_buf[sv_offset + k];
            float y = prod - dot_c;
            float t = dot_val + y;
            dot_c = (t - dot_val) - y;
            dot_val = t;
        }
        
        float term = coef_buf[coef_base_j + sv_idx] * dot_val;
        float y = term - c2;
        float t = s2 + y;
        c2 = (t - s2) - y;
        s2 = t;
    }
    
    score_buf[sample_idx * num_pairs + pair_idx] = rho_buf[pair_idx] + s1 + s2;
}
""")
        self.poly_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer SVBuf { float sv_buf[]; };
layout(set = 0, binding = 2) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 3) readonly buffer RhoBuf { float rho_buf[]; };
layout(set = 0, binding = 4) readonly buffer StartVecBuf { float start_vec_buf[]; };
layout(set = 0, binding = 5) readonly buffer VecPerClassBuf { float vec_per_class_buf[]; };
layout(set = 0, binding = 6) writeonly buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float class_count_f = 0.0;
layout(constant_id = 3) const float vector_count_f = 0.0;
layout(constant_id = 4) const float gamma_f = 0.0;
layout(constant_id = 5) const float coef0_f = 0.0;
layout(constant_id = 6) const float degree_f = 0.0;
layout(constant_id = 7) const float num_pairs_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint pair_idx = gl_GlobalInvocationID.y;
    uint feature_size = uint(feature_size_f);
    uint class_count = uint(class_count_f);
    uint vector_count = uint(vector_count_f);
    uint degree = uint(degree_f);
    uint num_pairs = uint(num_pairs_f);

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
    
    float s1 = 0.0;
    float c1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        float dot_c = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float prod = in_buf[in_offset + k] * sv_buf[sv_offset + k];
            float y = prod - dot_c;
            float t = dot_val + y;
            dot_c = (t - dot_val) - y;
            dot_val = t;
        }
        
        float kernel_val = dot_val * gamma_f + coef0_f;
        float powered = kernel_val;
        for (uint d = 1; d < degree; ++d) {
            powered *= kernel_val;
        }
        
        float term = coef_buf[coef_base_i + sv_idx] * powered;
        float y = term - c1;
        float t = s1 + y;
        c1 = (t - s1) - y;
        s1 = t;
    }
    
    float s2 = 0.0;
    float c2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        float dot_c = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float prod = in_buf[in_offset + k] * sv_buf[sv_offset + k];
            float y = prod - dot_c;
            float t = dot_val + y;
            dot_c = (t - dot_val) - y;
            dot_val = t;
        }
        
        float kernel_val = dot_val * gamma_f + coef0_f;
        float powered = kernel_val;
        for (uint d = 1; d < degree; ++d) {
            powered *= kernel_val;
        }
        
        float term = coef_buf[coef_base_j + sv_idx] * powered;
        float y = term - c2;
        float t = s2 + y;
        c2 = (t - s2) - y;
        s2 = t;
    }
    
    score_buf[sample_idx * num_pairs + pair_idx] = rho_buf[pair_idx] + s1 + s2;
}
""")
        self.rbf_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer SVBuf { float sv_buf[]; };
layout(set = 0, binding = 2) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 3) readonly buffer RhoBuf { float rho_buf[]; };
layout(set = 0, binding = 4) readonly buffer StartVecBuf { float start_vec_buf[]; };
layout(set = 0, binding = 5) readonly buffer VecPerClassBuf { float vec_per_class_buf[]; };
layout(set = 0, binding = 6) writeonly buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float class_count_f = 0.0;
layout(constant_id = 3) const float vector_count_f = 0.0;
layout(constant_id = 4) const float gamma_f = 0.0;
layout(constant_id = 5) const float num_pairs_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint pair_idx = gl_GlobalInvocationID.y;
    uint feature_size = uint(feature_size_f);
    uint class_count = uint(class_count_f);
    uint vector_count = uint(vector_count_f);
    uint num_pairs = uint(num_pairs_f);
    
    float neg_gamma = -gamma_f;
    
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
    
    float s1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float squared_dist = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float diff = in_buf[in_offset + k] - sv_buf[sv_offset + k];
            squared_dist += diff * diff;
        }
        
        s1 += coef_buf[coef_base_i + sv_idx] * exp(neg_gamma * squared_dist);
    }
    
    float s2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float squared_dist = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float diff = in_buf[in_offset + k] - sv_buf[sv_offset + k];
            squared_dist += diff * diff;
        }
        
        s2 += coef_buf[coef_base_j + sv_idx] * exp(neg_gamma * squared_dist);
    }
    
    score_buf[sample_idx * num_pairs + pair_idx] = rho_buf[pair_idx] + s1 + s2;
}
""")
        self.sigmoid_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer SVBuf { float sv_buf[]; };
layout(set = 0, binding = 2) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 3) readonly buffer RhoBuf { float rho_buf[]; };
layout(set = 0, binding = 4) readonly buffer StartVecBuf { float start_vec_buf[]; };
layout(set = 0, binding = 5) readonly buffer VecPerClassBuf { float vec_per_class_buf[]; };
layout(set = 0, binding = 6) writeonly buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float class_count_f = 0.0;
layout(constant_id = 3) const float vector_count_f = 0.0;
layout(constant_id = 4) const float gamma_f = 0.0;
layout(constant_id = 5) const float coef0_f = 0.0;
layout(constant_id = 6) const float num_pairs_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint pair_idx = gl_GlobalInvocationID.y;
    uint feature_size = uint(feature_size_f);
    uint class_count = uint(class_count_f);
    uint vector_count = uint(vector_count_f);
    uint num_pairs = uint(num_pairs_f);
    
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
    
    float s1 = 0.0;
    float c1 = 0.0;
    for (uint vi = 0; vi < class_i_sc; ++vi) {
        uint sv_idx = si_i + vi;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        float dot_c = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float prod = in_buf[in_offset + k] * sv_buf[sv_offset + k];
            float y = prod - dot_c;
            float t = dot_val + y;
            dot_c = (t - dot_val) - y;
            dot_val = t;
        }
        
        float term = coef_buf[coef_base_i + sv_idx] * tanh(dot_val * gamma_f + coef0_f);
        float y = term - c1;
        float t = s1 + y;
        c1 = (t - s1) - y;
        s1 = t;
    }
    
    float s2 = 0.0;
    float c2 = 0.0;
    for (uint vj = 0; vj < class_j_sc; ++vj) {
        uint sv_idx = si_j + vj;
        uint sv_offset = sv_idx * feature_size;
        
        float dot_val = 0.0;
        float dot_c = 0.0;
        for (uint k = 0; k < feature_size; ++k) {
            float prod = in_buf[in_offset + k] * sv_buf[sv_offset + k];
            float y = prod - dot_c;
            float t = dot_val + y;
            dot_c = (t - dot_val) - y;
            dot_val = t;
        }
        
        float term = coef_buf[coef_base_j + sv_idx] * tanh(dot_val * gamma_f + coef0_f);
        float y = term - c2;
        float t = s2 + y;
        c2 = (t - s2) - y;
        s2 = t;
    }
    
    score_buf[sample_idx * num_pairs + pair_idx] = rho_buf[pair_idx] + s1 + s2;
}
""")
        self.vote_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) readonly buffer ScoreBuf { float score_buf[]; };
layout(set = 0, binding = 1) writeonly buffer VoteBuf { float vote_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float class_count_f = 0.0;
layout(constant_id = 2) const float num_pairs_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint class_count = uint(class_count_f);
    uint num_pairs = uint(num_pairs_f);
    
    uint vote_base = sample_idx * class_count;
    
    // Initialize votes to 0 for this sample
    for (uint c = 0; c < class_count; ++c) {
        vote_buf[vote_base + c] = 0.0;
    }
    
    // Compute votes from scores
    uint score_base = sample_idx * num_pairs;
    uint pair_idx = 0;
    
    for (uint i = 0; i < class_count - 1; ++i) {
        for (uint j = i + 1; j < class_count; ++j) {
            if (score_buf[score_base + pair_idx] > 0.0) {
                vote_buf[vote_base + i] += 1.0;
            } else {
                vote_buf[vote_base + j] += 1.0;
            }
            ++pair_idx;
        }
    }
}
""")
        self.softmax_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float num_classes_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint num_classes = uint(num_classes_f);
    
    uint base_idx = sample_idx * num_classes;
    
    // Find max value
    float v_max = score_buf[base_idx];
    for (uint i = 1; i < num_classes; ++i) {
        float val = score_buf[base_idx + i];
        if (val > v_max) {
            v_max = val;
        }
    }
    
    // Compute exp(val - v_max) and sum
    float sum_exp = 0.0;
    for (uint i = 0; i < num_classes; ++i) {
        uint idx = base_idx + i;
        score_buf[idx] = exp(score_buf[idx] - v_max);
        sum_exp += score_buf[idx];
    }
    
    // Normalize
    for (uint i = 0; i < num_classes; ++i) {
        score_buf[base_idx + i] /= sum_exp;
    }
}
""")
        self.logistic_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float num_classes_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint num_classes = uint(num_classes_f);
    
    uint base_idx = sample_idx * num_classes;
    
    for (uint i = 0; i < num_classes; ++i) {
        uint idx = base_idx + i;
        float val = score_buf[idx];
        float abs_val = abs(val);
        float v = 1.0 / (1.0 + exp(-abs_val));
        score_buf[idx] = (val < 0.0) ? (1.0 - v) : v;
    }
}
""")
        self.softmax_zero_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float num_classes_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint num_classes = uint(num_classes_f);
    
    uint base_idx = sample_idx * num_classes;
    
    // Find max value
    float v_max = score_buf[base_idx];
    for (uint i = 1; i < num_classes; ++i) {
        float val = score_buf[base_idx + i];
        if (val > v_max) {
            v_max = val;
        }
    }
    
    float exp_neg_v_max = exp(-v_max);
    float sum_val = 0.0;
    
    for (uint i = 0; i < num_classes; ++i) {
        uint idx = base_idx + i;
        float v = score_buf[idx];
        if (v > 0.0000001 || v < -0.0000001) {
            score_buf[idx] = exp(v - v_max);
        } else {
            score_buf[idx] = v * exp_neg_v_max;
        }
        sum_val += score_buf[idx];
    }
    
    // Normalize
    float norm_val = (sum_val == 0.0) ? 0.5 : (1.0 / sum_val);
    for (uint i = 0; i < num_classes; ++i) {
        score_buf[base_idx + i] *= norm_val;
    }
}
""")
        self.probit_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) buffer ScoreBuf { float score_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float num_classes_f = 0.0;

const float PI = 3.14159265358979323846;
const float SQRT2 = 1.41421356;

float erf_inv(float x) {
    float sgn = (x < 0.0) ? -1.0 : 1.0;
    x = (1.0 - x) * (1.0 + x);
    if (x == 0.0) return 0.0;
    
    float log_val = log(x);
    float v = 2.0 / (PI * 0.147) + 0.5 * log_val;
    float v2 = (1.0 / 0.147) * log_val;
    float v3 = -v + sqrt(v * v - v2);
    return sgn * sqrt(v3);
}

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint num_classes = uint(num_classes_f);
    
    uint base_idx = sample_idx * num_classes;
    
    for (uint i = 0; i < num_classes; ++i) {
        uint idx = base_idx + i;
        float val = score_buf[idx];
        score_buf[idx] = SQRT2 * erf_inv(val * 2.0 - 1.0);
    }
}
""")
        self.argmax_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf { float in_data[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { uint out_data[]; };

layout(constant_id = 0) const float axis_size_f = 0.0;
layout(constant_id = 1) const float block_size_f = 0.0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    uint axis_size = uint(axis_size_f);
    uint block_size = uint(block_size_f);
    
    uint base_idx = gx * axis_size * block_size + gy;
    
    float max_val = in_data[base_idx];
    uint max_idx = 0u;
    base_idx += block_size;
    
    for (uint i = 1u; i < axis_size; ++i, base_idx += block_size) {
        if (in_data[base_idx] > max_val) {
            max_val = in_data[base_idx];
            max_idx = i;
        }
    }
    out_data[gx * block_size + gy] = max_idx;
}
""")
        self.sigmoid_prob_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(set = 0, binding = 0) buffer ScoreBuf { float score_buf[]; };
layout(set = 0, binding = 1) readonly buffer ProbABuf { float prob_a_buf[]; };
layout(set = 0, binding = 2) readonly buffer ProbBBuf { float prob_b_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float num_pairs_f = 0.0;

void main() {
    uint sample_idx = gl_GlobalInvocationID.x;
    uint pair_idx = gl_GlobalInvocationID.y;
    uint num_pairs = uint(num_pairs_f);
    
    uint idx = sample_idx * num_pairs + pair_idx;
    float score = score_buf[idx];
    float val = score * prob_a_buf[pair_idx] + prob_b_buf[pair_idx];
    float abs_val = abs(val);
    float logistic_val = 1.0 / (1.0 + exp(-abs_val));
    logistic_val = (val < 0.0) ? (1.0 - logistic_val) : logistic_val;
    score_buf[idx] = 1.0 - logistic_val;
}
""")
        self.label_map_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) readonly buffer IndicesBuf { uint indices_buf[]; };
layout(set = 0, binding = 1) readonly buffer LabelsBuf { float labels_buf[]; };
layout(set = 0, binding = 2) writeonly buffer OutBuf { float out_buf[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint label_idx = indices_buf[idx];
    out_buf[idx] = labels_buf[label_idx];
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"SVMClassifierOp({dev})"

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
        feature_size = np.prod(in_shape[1:])

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
            start_vec_tensor = self.manager.tensor(starting_vector)
            vec_per_class_tensor = self.manager.tensor(self.vectors_per_class)
            score_tensor = self.manager.tensor(np.zeros(n_samples * num_pairs, dtype=np.float32))

            updated_tensors.extend([
                sv_tensor, coef_tensor, rho_tensor, start_vec_tensor, vec_per_class_tensor, score_tensor
            ])

            # Step 1: Compute scores
            if kernel_upper == "LINEAR":
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, sv_tensor, coef_tensor, rho_tensor, start_vec_tensor, vec_per_class_tensor, score_tensor],
                    self.svc_linear_shader,
                    (n_samples, num_pairs, 1),
                    [n_samples, feature_size, class_count, vector_count, num_pairs]
                ))
            elif kernel_upper == "POLY":
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, sv_tensor, coef_tensor, rho_tensor, start_vec_tensor, vec_per_class_tensor, score_tensor],
                    self.poly_shader,
                    (n_samples, num_pairs, 1),
                    [n_samples, feature_size, class_count, vector_count, gamma, coef0, degree, num_pairs]
                ))
            elif kernel_upper == "RBF":
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, sv_tensor, coef_tensor, rho_tensor, start_vec_tensor, vec_per_class_tensor, score_tensor],
                    self.rbf_shader,
                    (n_samples, num_pairs, 1),
                    [n_samples, feature_size, class_count, vector_count, gamma, num_pairs]
                ))
            elif kernel_upper == "SIGMOID":
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, sv_tensor, coef_tensor, rho_tensor, start_vec_tensor, vec_per_class_tensor, score_tensor],
                    self.sigmoid_shader,
                    (n_samples, num_pairs, 1),
                    [n_samples, feature_size, class_count, vector_count, gamma, coef0, num_pairs]
                ))
            else:
                raise ValueError(f"Unsupported kernel type: {kernel_upper}")

            # Step 1.5: Apply probability transformation if has_proba
            if has_proba:
                prob_a_tensor = self.manager.tensor(self.prob_a)
                prob_b_tensor = self.manager.tensor(self.prob_b)
                updated_tensors.extend([prob_a_tensor, prob_b_tensor])

                updated_algorithms.append(
                    self.manager.algorithm(
                        [score_tensor, prob_a_tensor, prob_b_tensor],
                        self.sigmoid_prob_shader,
                        (n_samples, num_pairs, 1),
                        [n_samples, num_pairs]
                    )
                )


            # Step 2: Compute votes from scores (GPU implementation)
            vote_tensor = self.manager.tensor(np.zeros(n_samples * class_count, dtype=np.float32))

            updated_algorithms.append(
                self.manager.algorithm(
                    [score_tensor, vote_tensor],
                    self.vote_shader,
                    (n_samples, 1, 1),
                    [n_samples, class_count, num_pairs]
                )
            )

            # Step 3: Argmax on votes to get label indices
            label_indices_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.uint32))

            updated_algorithms.append(
                self.manager.algorithm(
                    [vote_tensor, label_indices_tensor],
                    self.argmax_shader,
                    (n_samples, 1, 1),
                    [class_count, 1.0]
                )
            )

            # Map indices to actual labels
            if self.classlabels_strings is not None or classlabels is None or len(classlabels) == 0:
                labels_lut_data = [i for i in range(class_count)]
            else:
                labels_lut_data = classlabels

            labels_lut_tensor = self.manager.tensor(labels_lut_data)
            labels_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            # labels_lut_tensor需要同步数据，labels_tensor和score_tensor是最终输出
            updated_tensors.extend([labels_lut_tensor, labels_tensor, score_tensor])

            updated_algorithms.append(
                self.manager.algorithm(
                    [label_indices_tensor, labels_lut_tensor, labels_tensor],
                    self.label_map_shader,
                    (n_samples, 1, 1),
                    []
                )
            )

            # 设置用于post_transform的tensor和相关参数
            transform_tensor = vote_tensor
            output_score_shape = (n_samples, num_pairs)

        else:
            # SVM_LINEAR mode: vector_count == 0
            # Formula: for each class j, score[j] = X · coefficients[j] + rho[0]
            # Output: (n_samples, class_count) scores

            # coefficients shape: (class_count, feature_size)
            coef_data = self.coefficients.astype(np.float32).reshape(-1)
            rho0 = self.rho[0] if self.rho is not None and len(self.rho) > 0 else 0.0

            coef_tensor = self.manager.tensor(coef_data)
            score_tensor = self.manager.tensor(np.zeros(n_samples * class_count, dtype=np.float32))
            # coef_tensor需要同步数据到GPU
            updated_tensors.append(coef_tensor)

            # Use linear_shader: compute X · coefficients[j] + rho[0] for each class j
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, coef_tensor, score_tensor],
                self.linear_shader,
                (n_samples * class_count, 1, 1),
                [n_samples, feature_size, class_count, rho0]
            ))

            # In LINEAR mode, scores are (n_samples, class_count), not pairs
            # We need to use argmax directly on scores, not votes
            label_indices_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.uint32))
            # label_indices_tensor是中间缓冲区，不需要同步数据到GPU

            updated_algorithms.append(
                self.manager.algorithm(
                    [score_tensor, label_indices_tensor],
                    self.argmax_shader,
                    (n_samples, 1, 1),
                    [class_count, 1.0]
                )
            )

            # Map indices to actual labels
            if self.classlabels_strings is not None or classlabels is None or len(classlabels) == 0:
                labels_lut_data = [i for i in range(class_count)]
            else:
                labels_lut_data = classlabels

            labels_lut_tensor = self.manager.tensor(labels_lut_data)
            labels_tensor = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            # labels_lut_tensor需要同步数据，labels_tensor和score_tensor是最终输出
            updated_tensors.extend([labels_lut_tensor, labels_tensor, score_tensor])

            updated_algorithms.append(
                self.manager.algorithm(
                    [label_indices_tensor, labels_lut_tensor, labels_tensor],
                    self.label_map_shader,
                    (n_samples, 1, 1),
                    []
                )
            )

            # 设置用于post_transform的tensor和相关参数
            transform_tensor = score_tensor
            output_score_shape = (n_samples, class_count)

        # 应用 post_transform（两个分支共享）
        if post_transform_upper != "NONE":
            if post_transform_upper == "SOFTMAX":
                updated_algorithms.append(
                    self.manager.algorithm(
                        [transform_tensor],
                        self.softmax_shader,
                        (n_samples, 1, 1),
                        [n_samples, class_count]
                    )
                )
            elif post_transform_upper == "LOGISTIC":
                updated_algorithms.append(
                    self.manager.algorithm(
                        [transform_tensor],
                        self.logistic_shader,
                        (n_samples, 1, 1),
                        [n_samples, class_count]
                    )
                )
            elif post_transform_upper == "SOFTMAX_ZERO":
                updated_algorithms.append(
                    self.manager.algorithm(
                        [transform_tensor],
                        self.softmax_zero_shader,
                        (n_samples, 1, 1),
                        [n_samples, class_count]
                    )
                )
            elif post_transform_upper == "PROBIT":
                updated_algorithms.append(
                    self.manager.algorithm(
                        [transform_tensor],
                        self.probit_shader,
                        (n_samples, 1, 1),
                        [n_samples, class_count]
                    )
                )

        return [
            (labels_tensor, (n_samples,)),
            (score_tensor, output_score_shape)
        ]
