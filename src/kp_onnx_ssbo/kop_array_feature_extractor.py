import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class ArrayFeatureExtractorOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer buf_data {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer buf_indices {{ int indices[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_output {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

layout (constant_id = 0) const float last_dim_f = 0;
layout (constant_id = 1) const float num_indices_f = 0;

void main() 
{{
    uint left_size = params[0], num_indices = params[1], last_dim = params[2];
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    if(gx >= left_size || gy >= num_indices) return;
    
    int col_idx = indices[gy];
    if (col_idx < 0)
        col_idx += int(last_dim);
    
    uint data_idx = gx * last_dim + uint(col_idx);
    uint output_idx = gx * num_indices + gy;
    
    out_tensor[output_idx] = in_tensor[data_idx];
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ArrayFeatureExtractorOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        data = inputs[0]
        indices = inputs[1]

        numpy_in_data = data.reshape(-1).astype(np.float32) \
            if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
        tensor_data = self.manager.tensor(numpy_in_data)
        input_tensors.append((tensor_data, list(data.shape) if isinstance(data, np.ndarray) else [len(data)]))

        numpy_in_indices = indices.reshape(-1).astype(np.int32) \
            if isinstance(indices, np.ndarray) else np.array(indices, dtype=np.int32)
        tensor_indices = self.manager.tensor_t(numpy_in_indices)
        input_tensors.append((tensor_indices, list(indices.shape) if isinstance(indices, np.ndarray) else [len(indices)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

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
        tensor_data, shape_data = input_tensors[0]
        tensor_indices, shape_indices = input_tensors[1]

        num_indices = int(np.prod(shape_indices))

        left_size = int(np.prod(shape_data[:-1]))
        last_dim = shape_data[-1]
        new_shape = shape_data[:-1] + [num_indices]

        total_output_size = left_size * num_indices
        tensor_out = self.manager.tensor(np.zeros(total_output_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params = [left_size, num_indices, last_dim]
        param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
        group_x = (left_size + LOCAL_X_2D - 1) // LOCAL_X_2D
        group_y = (num_indices + LOCAL_Y_2D - 1) // LOCAL_Y_2D
        workgroup = (group_x, group_y, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_data, tensor_indices, tensor_out, param_in],
            self.compiled_shader,
            workgroup
        ))
        
        return [(tensor_out, new_shape)]
