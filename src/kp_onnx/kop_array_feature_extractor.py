import numpy as np
import kp
from .shader_utils import compile_source


class ArrayFeatureExtractorOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        
        self.compiled_shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_data { float data[]; };
layout (binding = 1) readonly buffer buf_indices { int indices[]; };
layout (binding = 2) buffer buf_output { float tensor_out[]; };

layout (constant_id = 0) const float last_dim_f = 0;
layout (constant_id = 1) const float num_indices_f = 0;

void main() {
    uint left_idx = gl_GlobalInvocationID.x;
    uint index_idx = gl_GlobalInvocationID.y;
    
    uint last_dim = uint(last_dim_f);
    uint num_indices = uint(num_indices_f);
    
    int col_idx = indices[index_idx];
    if (col_idx < 0) {
        col_idx += int(last_dim);
    }
    
    uint data_idx = left_idx * last_dim + uint(col_idx);
    uint output_idx = left_idx * num_indices + index_idx;
    
    tensor_out[output_idx] = data[data_idx];
}
""")

    def __repr__(self):
        return f"ArrayFeatureExtractorOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

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

        updated_algorithms.append(self.manager.algorithm(
            [tensor_data, tensor_indices, tensor_out],
            self.compiled_shader,
            (left_size, num_indices, 1),
            [last_dim, num_indices],
            []
        ))
        
        return [(tensor_out, new_shape)]
