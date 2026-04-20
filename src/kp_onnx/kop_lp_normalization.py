import numpy as np
import kp
from .shader_utils import compile_source


class LpNormalizationOp:
    def __init__(self, manager: kp.Manager, axis=-1, p=2):
        self.manager = manager
        self.axis = axis
        self.p = p
        self.compiled_shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) readonly  buffer buf_in_tensor  { float in_tensor[];  };
layout (binding = 1) writeonly buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float dimension_f = 0;
layout (constant_id = 1) const float block_size_f = 0;
layout (constant_id = 2) const float p_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);

    uint in_index = gx * dimension * block_size + gy;
    uint mid_index = gx * block_size + gy;

    float pow_sum = 0.0;
    for (uint i = 0; i < dimension; ++i, in_index += block_size) {
        pow_sum += pow(in_tensor[in_index], p_f);
    }

    float pow_sum_pow = pow(pow_sum, 1.0 / p_f);
    in_index = gx * dimension * block_size + gy;
    for (uint i = 0; i < dimension; ++i, in_index += block_size) {
        out_tensor[in_index] = in_tensor[in_index] / pow_sum_pow;
    }
}""")

    def __repr__(self):
        return f"LpNormalizationOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors = [(tensor, list(inputs[0].shape))]

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]]))
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
        axis = self.axis
        if axis < 0:
            axis += len(shape_in)
        p = self.p

        dimension = shape_in[axis]
        group_x = int(np.prod(shape_in[:axis])) if axis > 0 else 1
        block_size = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1

        tensor_out = self.manager.tensor(np.zeros(group_x * dimension * block_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out],
                                                         self.compiled_shader,
                                                         (group_x, block_size, 1),
                                                         [dimension, block_size, p],
                                                         []))
        return [(tensor_out, shape_in)]
