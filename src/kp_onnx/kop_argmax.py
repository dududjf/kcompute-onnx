import numpy as np
import kp
from .shader_utils import compile_source


class ArgMaxOp:
    def __init__(self, manager: kp.Manager, axis=0, keepdims=True, select_last_index=False):
        self.manager = manager
        self.axis = axis
        self.keepdims = keepdims
        self.select_last_index = select_last_index
        self.shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(set=0, binding=0) buffer InBuf  { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf { uint out_data[];    };

layout(constant_id = 0) const float axis_size_f = 0;
layout(constant_id = 1) const float block_size_f = 0;
layout(constant_id = 2) const float select_last_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint axis_size = uint(axis_size_f);
    uint block_size = uint(block_size_f);
    uint select_last = uint(select_last_f);

    uint base_idx = gx * axis_size * block_size + gy;

    float max_val = in_data[base_idx];
    uint max_idx = 0u;
    base_idx += block_size;

    if (select_last == 1u) {
        for (uint i = 1u; i < axis_size; ++i, base_idx += block_size) {
            if (in_data[base_idx] >= max_val) {
                max_val = in_data[base_idx];
                max_idx = i;
            }
        }
    } 
    else {
        for (uint i = 1u; i < axis_size; ++i, base_idx += block_size) {
            if (in_data[base_idx] > max_val) {
                max_val = in_data[base_idx];
                max_idx = i;
            }
        }
    }
    out_data[gx * block_size + gy] = max_idx;
}
""")

    def __repr__(self):
        return f"ArgMaxOp({self.manager.get_device_properties()['device_name']})"

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

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape).astype(np.int64)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        axis = self.axis

        axis += len(shape_in) if axis < 0 else 0

        axis_size = shape_in[axis]
        batch_size = int(np.prod(shape_in[:axis])) if axis >= 0 else 1
        block_size = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1

        tensor_out = self.manager.tensor_t(np.zeros(np.prod(batch_size * block_size), dtype=np.int32))
        updated_tensors.append(tensor_out)

        workgroup = (batch_size, block_size, 1)

        spec_consts = [axis_size, block_size, self.select_last_index]
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            spec_consts,
            []
        ))

        if self.keepdims:
            output_shape = shape_in[:axis] + [1] + shape_in[axis + 1:]
        else:
            output_shape = shape_in[:axis] + shape_in[axis + 1:]

        return [(tensor_out, output_shape)]