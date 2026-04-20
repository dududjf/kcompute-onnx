import numpy as np
import kp
from .shader_utils import compile_source


class SoftmaxOp:
    def __init__(self, manager: kp.Manager, axis=-1):
        self.axis = axis
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(set=0, binding=0) buffer InBuf  { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float axis_size_f = 0;
layout(constant_id = 1) const float block_size_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    uint axis_size  = uint(axis_size_f);
    uint block_size = uint(block_size_f);
    
    uint base = gx * axis_size * block_size + gy;
    
    uint  idx = base;
    float max_val = in_data[idx];
    for (uint i = 1u; i < axis_size; ++i) {
        idx += block_size;
        float v = in_data[idx];
        max_val = max(max_val, v);
    }
    
    idx = base;
    float sum_exp = 0.0;
    for (uint i = 0u; i < axis_size; ++i) {
        float v = in_data[idx] - max_val;
        sum_exp += exp(v);
        idx += block_size;
    }
    
    float inv_sum = 1.0 / sum_exp;
    idx = base;
    for (uint i = 0u; i < axis_size; ++i) {
        float v = in_data[idx] - max_val;
        out_data[idx] = exp(v) * inv_sum;
        idx += block_size;
    }
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SoftmaxOp({device_name})"

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
        axis = self.axis if self.axis >= 0 else self.axis + len(shape_in)
        axis_size = shape_in[axis]

        group_x = int(np.prod(shape_in[:axis])) if axis >= 0 else 1
        block_size = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1
        total_size = int(np.prod(shape_in))

        tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (group_x, block_size, 1)

        spec_consts = [axis_size, block_size]
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            spec_consts,
            []
        ))

        return [(tensor_out, shape_in)]