import numpy as np
import kp
from .shader_utils import compile_source


class HardmaxOp:
    def __init__(self, manager: kp.Manager, axis=-1):
        self.manager = manager
        self.axis = axis
        self.shader = compile_source(r"""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer InBuf  { float in_data[];  };
layout (binding = 1) writeonly buffer OutBuf { float out_data[]; };

layout (constant_id = 0) const float axis_size_f  = 0.0;
layout (constant_id = 1) const float block_size_f = 0.0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint axis_size  = uint(axis_size_f);
    uint block_size = uint(block_size_f);

    uint initial_base = gx * axis_size * block_size + gy;
    float max_value = in_data[initial_base];
    uint max_idx = initial_base;
    
    uint base = initial_base;
    for (uint i = 0u; i < axis_size; ++i, base += block_size) {
        if (in_data[base] > max_value) {
            max_value = in_data[base];
            max_idx = base;
        }
    }

    base = initial_base;
    for (uint i = 0u; i < axis_size; ++i, base += block_size) {
        out_data[base] = base == max_idx ? 1.0 : 0.0;
    }
}
""")

    def __repr__(self):
        return f"HardmaxOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

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
        tensor_in, shape_in = input_tensors[0]

        self.axis += len(shape_in) if self.axis < 0 else 0

        axis_size = int(shape_in[self.axis])
        group_x = int(np.prod(shape_in[:self.axis])) if self.axis >= 0 else 1
        block_size = int(np.prod(shape_in[self.axis + 1:])) if self.axis + 1 < len(shape_in) else 1

        tensor_out = self.manager.tensor(np.zeros(np.prod(shape_in), dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (group_x, block_size, 1)
        spec_consts = [axis_size, block_size]

        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out],
                self.shader,
                workgroup,
                spec_consts,
                [],
            )
        )

        return [(tensor_out, shape_in)]
