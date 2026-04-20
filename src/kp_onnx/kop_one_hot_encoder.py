import numpy as np
import kp
from .shader_utils import compile_source


class OneHotEncoderOp:
    def __init__(self, manager: kp.Manager, cats_int64s=None, cats_strings=None, zeros=1):
        self.manager = manager
        self.cats_int64s = cats_int64s
        self.cats_strings = cats_strings
        self.zeros = zeros
        self.compiled_shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_input { int input_data[]; };
layout (binding = 1) readonly buffer buf_categories { int categories[]; };
layout (binding = 2) buffer buf_output { float output_data[]; };

layout (constant_id = 0) const float num_categories_f = 0;

void main() {
    uint input_idx = gl_GlobalInvocationID.x;
    uint category_idx = gl_GlobalInvocationID.y;
    
    uint num_categories = uint(num_categories_f);
    
    uint output_idx = input_idx * num_categories + category_idx;
    
    int input_value = input_data[input_idx];
    int category_value = categories[category_idx];
    
    output_data[output_idx] = input_value == category_value ? 1.0 : 0.0;
}
""")

    def __repr__(self):
        return f"OneHotEncoderOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.int32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.int32)
            tensor = self.manager.tensor_t(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        tensors_to_sync = [t[0] for t in input_tensors] + updated_tensors[:-1]
        seq.record(kp.OpTensorSyncDevice(tensors_to_sync))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([t[0] for t in output_tensor_and_shape]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]

        assert len(shape_in) <= 2, f"This operator is not implemented for shape {shape_in}."
        assert self.cats_int64s is not None or self.cats_strings is not None, "No encoding was defined."

        if self.cats_int64s is not None:
            categories = np.array(self.cats_int64s, dtype=np.int32)
        else:
            # Convert strings to integers using hash
            assert False, "String encoding is not implemented yet."

        num_categories = len(categories)
        input_size = int(np.prod(shape_in))
        
        shape_out = shape_in[:] + [num_categories]
        total_output_size = input_size * num_categories
        
        tensor_categories = self.manager.tensor_t(categories)
        updated_tensors.append(tensor_categories)
        
        tensor_out = self.manager.tensor(np.zeros(total_output_size, dtype=np.float32))
        updated_tensors.append(tensor_out)
        
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_categories, tensor_out],
            self.compiled_shader,
            (input_size, num_categories, 1),
            [num_categories],
            []
        ))
        
        return [(tensor_out, shape_out)]
