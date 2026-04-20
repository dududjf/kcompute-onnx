import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_1D


class NormalizerOp:
    def __init__(self, manager: kp.Manager, norm='MAX'):
        self.manager = manager
        self.norm = norm
        self.compiled_shader_max = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float input_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float output_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint n_rows = params[0];
    uint n_cols = params[1];
    uint row_idx = gl_GlobalInvocationID.x;
    if (row_idx >= n_rows) return;
    uint offset = row_idx * n_cols;
    uint end = offset + n_cols;
    
    float max_val = 0.0;
    for (uint i = offset; i < end; ++i) {{
        float val = abs(input_buf[i]);
        max_val = max(max_val, val);
    }}
    
    if (max_val != 0.0) {{
        for (uint i = offset; i < end; ++i) {{
            output_buf[i] = input_buf[i] / max_val;
        }}
    }} else {{
        for (uint i = offset; i < end; ++i) {{
            output_buf[i] = 0.0;
        }}
    }}
}}
""")

        self.compiled_shader_l1 = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float input_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float output_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint n_rows = params[0];
    uint n_cols = params[1];
    uint row_idx = gl_GlobalInvocationID.x;
    if (row_idx >= n_rows) return;
    uint offset = row_idx * n_cols;
    uint end = offset + n_cols;
    
    float sum = 0.0;
    for (uint i = offset; i < end; ++i) {{
        sum += abs(input_buf[i]);
    }}
    
    if (sum != 0.0) {{
        for (uint i = offset; i < end; ++i) {{
            output_buf[i] = input_buf[i] / sum;
        }}
    }} else {{
        for (uint i = offset; i < end; ++i) {{
            output_buf[i] = 0.0;
        }}
    }}
}}
""")
        
        # Shader for L2 norm
        self.compiled_shader_l2 = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float input_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float output_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint n_rows = params[0];
    uint n_cols = params[1];
    uint row_idx = gl_GlobalInvocationID.x;
    if (row_idx >= n_rows) return;
    uint offset = row_idx * n_cols;
    uint end = offset + n_cols;
    
    float sum_sq = 0.0;
    for (uint i = offset; i < end; ++i) {{
        float val = input_buf[i];
        sum_sq += val * val;
    }}
    
    if (sum_sq != 0.0) {{
        float norm = sqrt(sum_sq);
        for (uint i = offset; i < end; ++i) {{
            output_buf[i] = input_buf[i] / norm;
        }}
    }} else {{
        for (uint i = offset; i < end; ++i) {{
            output_buf[i] = 0.0;
        }}
    }}
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"NormalizerOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(shape_out)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        assert len(shape_in) == 1 or len(shape_in) == 2, "Input tensor must be at 1D or 2D"

        if len(shape_in) == 1:
            n_rows = 1
            n_cols = shape_in[0]
            shape_in = [n_rows, n_cols]
        else:
            n_rows = shape_in[0]
            n_cols = shape_in[1]
            shape_in = [n_rows, n_cols]
        
        if self.norm == 'MAX':
            norm_shader = self.compiled_shader_max
        elif self.norm == 'L1':
            norm_shader = self.compiled_shader_l1
        elif self.norm == 'L2':
            norm_shader = self.compiled_shader_l2
        else:
            raise ValueError(f"Unexpected value for norm='{self.norm}'.")

        tensor_out = self.manager.tensor(np.zeros(n_rows * n_cols, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params = np.array([n_rows, n_cols], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        workgroup = ((n_rows + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out, param_in],
            norm_shader,
            workgroup,
        ))
        
        return [(tensor_out, shape_in)]
