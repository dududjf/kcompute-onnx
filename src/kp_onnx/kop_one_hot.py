import numpy as np
import kp
from .shader_utils import compile_source


class OneHotOp:
    def __init__(self, manager: kp.Manager, axis: int = -1):
        self.manager = manager
        self.axis = axis
        self.compiled_shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_indices { float indices_buf[]; };
layout (binding = 1) buffer buf_output { float output_buf[]; };

layout (constant_id = 0) const float depth_f = 0;
layout (constant_id = 1) const float right_size_f = 0;
layout (constant_id = 2) const float off_value_f = 0;
layout (constant_id = 3) const float on_value_f = 0;

void main() {
    uint left_idx = gl_GlobalInvocationID.x;
    uint depth_idx = gl_GlobalInvocationID.y;
    uint right_idx = gl_GlobalInvocationID.z;
    
    int depth = int(depth_f);
    uint right_size = uint(right_size_f);
    float off_value = off_value_f;
    float on_value = on_value_f;
    
    uint indices_idx = left_idx * right_size + right_idx;
    uint output_idx = (left_idx * depth + depth_idx) * right_size + right_idx;
    
    int index = int(indices_buf[indices_idx]);
    uint index_value = uint((index % depth + depth) % depth);
    
    if (depth_idx == index_value) {
        output_buf[output_idx] = on_value;
    } else {
        output_buf[output_idx] = off_value;
    }
}
""")

    def __repr__(self):
        return f"OneHotOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

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
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        if tensor_out is not None:
            output = tensor_out.data().reshape(output_shape)
        else:
            output = np.array([], dtype=np.float32).reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_indices, shape_indices = input_tensors[0]
        tensor_depth, shape_depth = input_tensors[1]
        tensor_values, shape_values = input_tensors[2]

        axis = self.axis
        if axis < 0:
            axis += len(shape_indices) + 1

        values = tensor_values.data()
        off_value, on_value = values[0], values[1]
        depth_value = int(tensor_depth.data()[0])

        if depth_value == 0:
            ls = shape_indices[0:axis]
            rs = shape_indices[axis:]
            shape_out = ls[:] + [depth_value] + rs[:]
            return [(None, shape_out)]

        else:
            ls = shape_indices[0:axis]
            rs = shape_indices[axis:]
            left_size = int(np.prod(ls)) if len(ls) > 0 else 1
            right_size = int(np.prod(rs)) if len(rs) > 0 else 1

            shape_out = ls[:] + [depth_value] + rs[:]
            total_size = left_size * depth_value * right_size

            tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
            updated_tensors.append(tensor_out)

            updated_algorithms.append(self.manager.algorithm(
                [tensor_indices, tensor_out],
                self.compiled_shader,
                (left_size, depth_value, right_size),
                [depth_value, right_size, off_value, on_value],
                []
            ))

            return [(tensor_out, shape_out)]