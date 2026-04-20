import kp
import numpy as np
from .shader_utils import compile_source


class GlobalMaxPoolOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source(r"""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float block_f = 0.0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint block = uint(block_f);
    uint base = gx * block;

    float max_val = in_buf[base];
    for (uint i = 1; i < block; ++i, ++base) {
        max_val = max(max_val, in_buf[base]);
    }
    out_buf[gx] = max_val;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"GlobalMaxPoolOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

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
        tensor_in, in_shape = input_tensors[0]
        assert len(in_shape) >= 2, "GlobalMaxPool expects at least [N, C, ...]"

        spatial_shape = len(in_shape) - 2
        prefix_shape = in_shape[:-2]
        out_shape = prefix_shape + [1] * spatial_shape

        if len(in_shape) == 2:
            num_prefix = 1
            H = in_shape[0]
            W = in_shape[1]
        else:
            num_prefix = int(np.prod(prefix_shape, dtype=np.int64)) if len(prefix_shape) > 0 else 1
            H = in_shape[-2]
            W = in_shape[-1]

        tensor_out = self.manager.tensor(np.zeros(num_prefix, dtype=np.float32))
        block = int(H) * int(W)
        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out],
                self.shader,
                (num_prefix, 1, 1),
                [block],
                [],
            )
        )
        updated_tensors.append(tensor_out)
        return [(tensor_out, out_shape)]
