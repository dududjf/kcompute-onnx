import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D


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

        self.linear_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf  {{ float in_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer CoefBuf {{ float coef_buf[]; }};
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_buf[]; }};
layout(std430, set = 0, binding = 3) readonly buffer Params {{
    uint feature_size;
    uint num_samples;
    float rho0;
}};

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples) return;
    
    uint in_idx = idx * feature_size;
    float s = 0.0;
    
    for (uint k = 0; k < feature_size; ++k) {{
        s += in_buf[in_idx] * coef_buf[k];
        ++in_idx;
    }}
    
    out_buf[idx] = s + rho0;
}}
""")

        # Common layout for kernel shaders
        # 0: InBuf
        # 1: SVBuf
        # 2: CoefBuf
        # 3: OutBuf
        # 4: Params
        kernel_layout = f"""
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf  {{ float in_buf[]; }};
layout(std430, set = 0, binding = 1) readonly buffer SVBuf {{ float sv_buf[]; }};
layout(std430, set = 0, binding = 2) readonly buffer CoefBuf {{ float coef_buf[]; }};
layout(std430, set = 0, binding = 3) writeonly buffer OutBuf {{ float out_buf[]; }};
layout(std430, set = 0, binding = 4) readonly buffer Params {{
    uint feature_size;
    uint num_samples;
    uint n_supports;
    float gamma;
    float coef0;
    float degree;
    float rho0;
}};
"""

        self.poly_shader = compile_source(f"""
#version 450
{kernel_layout}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples) return;
    
    uint base_offset = idx * feature_size;
    float s = 0.0;
    uint degree_int = uint(degree);

    uint sv_offset = 0;
    for (uint j = 0; j < n_supports; ++j) {{
        float dot_val = 0.0;
        uint in_idx = base_offset;
        uint sv_idx = sv_offset;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_idx] * sv_buf[sv_idx];
            in_idx++;
            sv_idx++;
        }}
        float kernel_val = dot_val * gamma + coef0; 
        float powered = 1.0;
        for (uint d = 0; d < degree_int; ++d) {{
            powered *= kernel_val;
        }}
        s += coef_buf[j] * powered;
        sv_offset += feature_size;
    }}
    s += rho0;
    
    out_buf[idx] = s;
}}
""")
        self.rbf_shader = compile_source(f"""
#version 450
{kernel_layout}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples) return;
    
    float neg_gamma = -gamma;
    
    uint base_offset = idx * feature_size;
    float s = 0.0;
    
    uint sv_offset = 0;
    for (uint j = 0; j < n_supports; ++j) {{
        float squared_dist = 0.0;
        uint in_idx = base_offset;
        uint sv_idx = sv_offset;
        for (uint k = 0; k < feature_size; ++k) {{
            float diff = in_buf[in_idx] - sv_buf[sv_idx];
            squared_dist += diff * diff;
            in_idx++;
            sv_idx++;
        }}
        float kernel_val = exp(neg_gamma * squared_dist);
        s += coef_buf[j] * kernel_val;
        sv_offset += feature_size;
    }}
    s += rho0;
    
    out_buf[idx] = s;
}}
""")
        self.sigmoid_shader = compile_source(f"""
#version 450
{kernel_layout}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples) return;
    
    uint base_offset = idx * feature_size;
    float s = 0.0;
    
    uint sv_offset = 0;
    for (uint j = 0; j < n_supports; ++j) {{
        float dot_val = 0.0;
        uint in_idx = base_offset;
        uint sv_idx = sv_offset;
        for (uint k = 0; k < feature_size; ++k) {{
            dot_val += in_buf[in_idx] * sv_buf[sv_idx];
            in_idx++;
            sv_idx++;
        }}
        float kernel_val = tanh(dot_val * gamma + coef0);
        s += coef_buf[j] * kernel_val;
        sv_offset += feature_size;
    }}
    s += rho0;
    
    out_buf[idx] = s;
}}
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

            # Add updated tensors that are NOT the outputs (if any)
            # Regressor usually creates one output tensor which is not synced from device (only to local)
            # But params tensors need sync.
            output_tensor_ids = {id(t) for t, _ in output_tensor_and_shape}
            for t in updated_tensors:
                if id(t) not in output_tensor_ids:
                    tensors_to_sync.append(t)

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
        feature_size = int(np.prod(in_shape[1:]))

        gamma, coef0, degree = (
            (self.kernel_params[0], self.kernel_params[1], self.kernel_params[2])
            if self.kernel_params else (0.0, 0.0, 0)
        )

        rho0 = self.rho[0] if self.rho else 0.0
        n_supports = self.n_supports if self.n_supports is not None else 0

        workgroup = ((n_samples + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

        if n_supports > 0:
            kernel_upper = self.kernel_type.upper()

            sv_data = self.support_vectors.astype(np.float32).reshape(-1)
            coef_data = self.coefficients.astype(np.float32).reshape(-1)

            sv_tensor = self.manager.tensor(sv_data)
            coef_tensor = self.manager.tensor(coef_data)
            tensor_out = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            updated_tensors.extend([sv_tensor, coef_tensor, tensor_out])

            # Params: feature_size, num_samples, n_supports, gamma, coef0, degree, rho0
            params = np.zeros(7, dtype=np.float32)
            params_uint = params.view(np.uint32)
            params_uint[0] = feature_size
            params_uint[1] = n_samples
            params_uint[2] = n_supports
            params[3] = gamma
            params[4] = coef0
            params[5] = float(degree)
            params[6] = rho0

            if kernel_upper == "LINEAR":
                params[3] = 1.0
                params[4] = 0.0
                params[5] = 1.0
                shader = self.poly_shader
            elif kernel_upper == "POLY":
                shader = self.poly_shader
            elif kernel_upper == "RBF":
                shader = self.rbf_shader
            elif kernel_upper == "SIGMOID":
                shader = self.sigmoid_shader
            else:
                raise ValueError(f"Unexpected kernel_type={self.kernel_type!r}.")

            params_tensor = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_tensor])).eval()

            tensors = [tensor_in, sv_tensor, coef_tensor, tensor_out, params_tensor]

            updated_algorithms.append(self.manager.algorithm(
                tensors,
                shader,
                workgroup
            ))
        else:
            coef_data = self.coefficients.astype(np.float32).reshape(-1)

            coef_tensor = self.manager.tensor(coef_data)
            tensor_out = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
            updated_tensors.extend([coef_tensor, tensor_out])

            # Params: feature_size, num_samples, rho0 (as float)
            params = np.zeros(3, dtype=np.float32)
            params_uint = params.view(np.uint32)
            params_uint[0] = feature_size
            params_uint[1] = n_samples
            params[2] = rho0

            params_tensor = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([params_tensor])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, coef_tensor, tensor_out, params_tensor],
                self.linear_shader,
                workgroup
            ))

        shape_out = [n_samples, 1]
        return [(tensor_out, shape_out)]
