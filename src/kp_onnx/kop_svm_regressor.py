import kp
import numpy as np
from .shader_utils import compile_source


class SVMRegressorOp:

    def __init__(self,
                 manager: kp.Manager,
                 coefficients=None,
                 kernel_params=None,
                 kernel_type="LINEAR",
                 n_targets=None,
                 n_supports=None,
                 one_class=None,
                 post_transform="NONE",
                 rho=None,
                 support_vectors=None):
        self.coefficients = coefficients
        self.kernel_params = kernel_params
        self.kernel_type = kernel_type
        self.n_targets = n_targets
        self.n_supports = n_supports
        self.one_class = one_class
        self.post_transform = post_transform
        self.rho = rho
        self.support_vectors = support_vectors
        self.manager = manager
        self.linear_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 2) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float rho0_f = 0.0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint feature_size = uint(feature_size_f);
    
    uint in_idx = idx * feature_size;
    float s = 0.0;
    
    for (uint k = 0; k < feature_size; ++k) {
        s += in_buf[in_idx] * coef_buf[k];
        ++in_idx;
    }
    
    out_buf[idx] = s + rho0_f;
}
""")
        self.poly_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer SVBuf { float sv_buf[]; };
layout(set = 0, binding = 2) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float n_supports_f = 0.0;
layout(constant_id = 3) const float gamma_f = 0.0;
layout(constant_id = 4) const float coef0_f = 0.0;
layout(constant_id = 5) const float degree_f = 0.0;
layout(constant_id = 6) const float rho0_f = 0.0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint n_samples = uint(n_samples_f);
    
    uint feature_size = uint(feature_size_f);
    uint n_supports = uint(n_supports_f);
    uint degree = uint(degree_f);
    
    uint base_offset = idx * feature_size;
    float s = 0.0;

    uint sv_offset = 0;
    for (uint j = 0; j < n_supports; ++j) {
        float dot_val = 0.0;
        uint in_idx = base_offset;
        uint sv_idx = sv_offset;
        for (uint k = 0; k < feature_size; ++k) {
            dot_val += in_buf[in_idx] * sv_buf[sv_idx];
            in_idx++;
            sv_idx++;
        }
        float kernel_val = dot_val * gamma_f + coef0_f; 
        float powered = 1.0;
        for (uint d = 0; d < degree; ++d) {
            powered *= kernel_val;
        }
        s += coef_buf[j] * powered;
        sv_offset += feature_size;
    }
    s += rho0_f;
    
    out_buf[idx] = s;
}
""")
        self.rbf_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer SVBuf { float sv_buf[]; };
layout(set = 0, binding = 2) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float n_supports_f = 0.0;
layout(constant_id = 3) const float gamma_f = 0.0;
layout(constant_id = 4) const float rho0_f = 0.0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint n_samples = uint(n_samples_f);
    
    uint feature_size = uint(feature_size_f);
    uint n_supports = uint(n_supports_f);
    
    uint base_offset = idx * feature_size;
    float s = 0.0;
    
    float neg_gamma = -gamma_f;
    
    uint sv_offset = 0;
    for (uint j = 0; j < n_supports; ++j) {
        float squared_dist = 0.0;
        uint in_idx = base_offset;
        uint sv_idx = sv_offset;
        for (uint k = 0; k < feature_size; ++k) {
            float diff = in_buf[in_idx] - sv_buf[sv_idx];
            squared_dist += diff * diff;
            in_idx++;
            sv_idx++;
        }
        float kernel_val = exp(neg_gamma * squared_dist);
        s += coef_buf[j] * kernel_val;
        sv_offset += feature_size;
    }
    s += rho0_f;
    
    out_buf[idx] = s;
}
""")
        self.sigmoid_shader = compile_source(r"""
#version 450
layout(local_size_x=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(set = 0, binding = 1) readonly buffer SVBuf { float sv_buf[]; };
layout(set = 0, binding = 2) readonly buffer CoefBuf { float coef_buf[]; };
layout(set = 0, binding = 3) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0) const float n_samples_f = 0.0;
layout(constant_id = 1) const float feature_size_f = 0.0;
layout(constant_id = 2) const float n_supports_f = 0.0;
layout(constant_id = 3) const float gamma_f = 0.0;
layout(constant_id = 4) const float coef0_f = 0.0;
layout(constant_id = 5) const float rho0_f = 0.0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint n_samples = uint(n_samples_f);
    
    uint feature_size = uint(feature_size_f);
    uint n_supports = uint(n_supports_f);
    
    uint base_offset = idx * feature_size;
    float s = 0.0;
    
    uint sv_offset = 0;
    for (uint j = 0; j < n_supports; ++j) {
        float dot_val = 0.0;
        uint in_idx = base_offset;
        uint sv_idx = sv_offset;
        for (uint k = 0; k < feature_size; ++k) {
            dot_val += in_buf[in_idx] * sv_buf[sv_idx];
            in_idx++;
            sv_idx++;
        }
        float kernel_val = tanh(dot_val * gamma_f + coef0_f);
        s += coef_buf[j] * kernel_val;
        sv_offset += feature_size;
    }
    s += rho0_f;
    
    out_buf[idx] = s;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"SVMRegressorOp({dev})"

    __str__ = __repr__

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
            tensors_to_sync = [t[0] for t in input_tensors]
            if len(updated_tensors) > 1:
                tensors_to_sync.extend(updated_tensors[:-1])
            seq.record(kp.OpTensorSyncDevice(tensors_to_sync))
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

        assert self.post_transform in (None, "NONE"), \
            f"post_transform={self.post_transform!r} not implemented. Only 'NONE' is supported."

        tensor_in, in_shape = input_tensors[0]
        n_samples = in_shape[0]
        feature_size = np.prod(in_shape[1:])

        gamma, coef0, degree = (
            (self.kernel_params[0], self.kernel_params[1], self.kernel_params[2])
            if self.kernel_params else (0.0, 0.0, 0)
        )

        rho0 = self.rho[0] if self.rho else 0.0
        n_supports = self.n_supports if self.n_supports is not None else 0
        workgroup = (n_samples, 1, 1)

        if n_supports > 0:
            kernel_upper = self.kernel_type.upper()

            sv_data = self.support_vectors.astype(np.float32).reshape(-1)
            coef_data = self.coefficients.astype(np.float32).reshape(-1)

            sv_tensor = self.manager.tensor(sv_data)
            coef_tensor = self.manager.tensor(coef_data)
            tensor_out = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            updated_tensors.extend([sv_tensor, coef_tensor, tensor_out])
            tensors = [tensor_in, sv_tensor, coef_tensor, tensor_out]

            if kernel_upper == "LINEAR":
                spec_consts = [
                    n_samples,
                    feature_size,
                    n_supports,
                    1.0,  # gamma
                    0.0,  # coef0
                    1.0,  # degree
                    rho0
                ]
                shader = self.poly_shader # Linear is a special case of Poly
            elif kernel_upper == "POLY":
                spec_consts = [
                    n_samples,
                    feature_size,
                    n_supports,
                    gamma,
                    coef0,
                    degree,
                    rho0
                ]
                shader = self.poly_shader
            elif kernel_upper == "RBF":
                spec_consts = [
                    n_samples,
                    feature_size,
                    n_supports,
                    gamma,
                    rho0
                ]
                shader = self.rbf_shader
            elif kernel_upper == "SIGMOID":
                spec_consts = [
                    n_samples,
                    feature_size,
                    n_supports,
                    gamma,
                    coef0,
                    rho0
                ]
                shader = self.sigmoid_shader
            else:
                raise ValueError(f"Unexpected kernel_type={self.kernel_type!r}.")

            updated_algorithms.append(self.manager.algorithm(
                tensors,
                shader,
                workgroup,
                spec_consts,
                []
            ))
        else:
            coef_data = self.coefficients.astype(np.float32).reshape(-1)

            coef_tensor = self.manager.tensor(coef_data)
            tensor_out = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            updated_tensors.extend([coef_tensor, tensor_out])

            spec_consts = [
                n_samples,
                feature_size,
                rho0
            ]
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, coef_tensor, tensor_out],
                self.linear_shader,
                workgroup,
                spec_consts,
                []
            ))

        shape_out = [n_samples, 1]
        return [(tensor_out, shape_out)]

