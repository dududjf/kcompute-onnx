import numpy as np
import kp
from .shader_utils import compile_source


class CompressOp:
    def __init__(self, manager: kp.Manager, axis=None):
        self.manager = manager
        self.axis = axis
        self.shader_no_axis = compile_source("""
#version 450
layout(local_size_x = 1) in;

layout(binding = 0) readonly  buffer in_buf     { float in_tensor[];     };
layout(binding = 1) readonly  buffer cond_buf   { uint  cond_tensor[];   };
layout(binding = 2) readonly  buffer prefix_buf { uint  prefix_tensor[]; };
layout(binding = 3) writeonly buffer out_buf    { float out_tensor[];    };

void main() {
    uint gx = gl_GlobalInvocationID.x;
    if (cond_tensor[gx] != 0) {
        uint out_index = prefix_tensor[gx];
        out_tensor[out_index] = in_tensor[gx];
    }
}
""")

        self.shader_axis = compile_source("""
#version 450
layout(local_size_x = 1) in;
layout(binding = 0) readonly  buffer in_buf     { float in_tensor[];     };
layout(binding = 1) readonly  buffer cond_buf   { uint  cond_tensor[];   };
layout(binding = 2) readonly  buffer prefix_buf { uint  prefix_tensor[]; };
layout(binding = 3) writeonly buffer out_buf    { float out_tensor[];    };

layout(constant_id = 0) const float dimension_f = 0;
layout(constant_id = 1) const float block_size_f = 0;
layout(constant_id = 2) const float out_block_size_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);
    uint out_block_size = uint(out_block_size_f);
    
    uint in_offset = gx * dimension * block_size + gy * block_size;
    uint out_offset = gx * out_block_size + prefix_tensor[gy] * block_size;
    
    if (cond_tensor[gy] != 0) {
        for (uint i = 0; i < block_size; i++, in_offset++, out_offset++) {
            out_tensor[out_offset] = in_tensor[in_offset];
        }
    }
}
""")

    def __repr__(self):
        return f"CompressOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        data_in = inputs[0].reshape(-1).astype(np.float32)
        data_shape = list(inputs[0].shape)
        condition = inputs[1].astype(np.int32)
        condition_shape = list(inputs[1].shape)
        data_tensor = self.manager.tensor(data_in)
        condition_tensor = self.manager.tensor_t(condition, tensor_type=kp.TensorTypes.device)
        input_tensors.append((data_tensor, data_shape))
        input_tensors.append((condition_tensor, condition_shape))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
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
        tensor_in, shape_in = input_tensors[0]
        tensor_condition, shape_condition = input_tensors[1]
        condition = tensor_condition.data()
        axis = self.axis

        if axis is not None:
            axis += len(shape_in) if axis < 0 else 0
            assert condition.size == shape_in[axis], "Condition size does not match input size"

            dimension = shape_in[axis]
            group_count = int(np.prod(shape_in[:axis])) if axis > 0 else 1
            block_size = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1

            prefix = np.zeros(np.prod(shape_condition), dtype=np.int32)
            running_sum = 0
            for i, c in enumerate(condition):
                prefix[i] = running_sum
                running_sum += int(c)
            kept = running_sum

            out_block_size = kept * block_size
            out_size = group_count * out_block_size
            out_array = np.zeros(out_size, dtype=np.float32)

            tensor_prefix = self.manager.tensor_t(prefix)
            tensor_out = self.manager.tensor(out_array)
            updated_tensors.extend([tensor_prefix, tensor_out])

            workgroup = (group_count, dimension, 1)
            updated_algorithms.append(
                self.manager.algorithm(
                    [tensor_in, tensor_condition, tensor_prefix, tensor_out],
                    self.shader_axis,
                    workgroup,
                    [dimension, block_size, out_block_size],
                    [],
                )
            )

            out_shape = list(shape_in)
            out_shape[axis] = kept
            return [(tensor_out, out_shape)]

        else:
            active_len = min(condition.size, np.prod(shape_in))
            prefix = np.zeros(active_len, dtype=np.int32)
            running_sum = 0
            for i, c in enumerate(condition):
                prefix[i] = running_sum
                running_sum += int(c)
            out_size = running_sum

            out_array = np.zeros(out_size, dtype=np.float32)
            tensor_prefix = self.manager.tensor_t(prefix)
            tensor_out = self.manager.tensor(out_array)

            updated_tensors.extend([tensor_prefix, tensor_out])
            workgroup = (active_len, 1, 1)
            updated_algorithms.append(
                self.manager.algorithm(
                    [tensor_in, tensor_condition, tensor_prefix, tensor_out],
                    self.shader_no_axis,
                    workgroup,
                    [],
                    [],
                )
            )
            return [(tensor_out, [out_size])]