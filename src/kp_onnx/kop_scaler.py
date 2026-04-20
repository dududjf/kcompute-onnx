import numpy as np
import kp
from .shader_utils import compile_source


class ScalerOp:
    def __init__(self, manager: kp.Manager, offset=None, scale=None):
        self.manager = manager
        self.offset = offset
        self.scale = scale

        self.compiled_shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) readonly  buffer buf_in     { float in_buf[];     };
layout (binding = 1) readonly  buffer buf_offset { float offset_buf[]; };
layout (binding = 2) readonly  buffer buf_scale  { float scale_buf[];  };
layout (binding = 3) writeonly buffer buf_out    { float out_buf[];    };

layout (constant_id = 0) const float trailing_f = 1.0;

void main() {
    uint leading_idx = gl_GlobalInvocationID.x;
    uint trailing_idx = gl_GlobalInvocationID.y;
    uint trailing = uint(trailing_f);
    
    uint idx = leading_idx * trailing + trailing_idx;
    out_buf[idx] = (in_buf[idx] - offset_buf[trailing_idx]) * scale_buf[trailing_idx];
}
""")

    def __repr__(self):
        return f"ScalerOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

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
        size = int(np.prod(shape_in))

        offset = self.offset
        scale = self.scale

        assert len(offset) == len(scale), "offset and scale must have the same length"

        trailing = len(offset)
        leading = size // trailing
        
        tensor_offset = self.manager.tensor(offset)
        tensor_scale = self.manager.tensor(scale)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.extend([tensor_offset, tensor_scale, tensor_out])
        
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_offset, tensor_scale, tensor_out],
            self.compiled_shader,
            (leading, trailing, 1),
            [trailing],
            []
        ))
        return [(tensor_out, list(shape_in))]
