import kp
import numpy as np
from .shader_utils import compile_source


class TransposeOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source("""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(binding=0) readonly  buffer InBuf  { float in_buf[];  };
layout(binding=1) writeonly buffer OutBuf { float out_buf[]; };
layout(constant_id = 0) const float pre_block_size_f = 0;
layout(constant_id = 1) const float post_block_size_f = 0;
layout(constant_id = 2) const float axis_dimension_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.y;
    uint gy = gl_GlobalInvocationID.x;
    uint gz = gl_GlobalInvocationID.z;

    uint pre_block_size = uint(pre_block_size_f);
    uint post_block_size = uint(post_block_size_f);
    uint axis_dimension = uint(axis_dimension_f);

    uint stride_y = axis_dimension * post_block_size;
    uint stride_x = pre_block_size * stride_y;
    uint in_index = gx * stride_x + gy * stride_y + gz * post_block_size;
    uint out_index = gx * stride_x + gz * pre_block_size * post_block_size + gy * post_block_size;

    for(uint i = 0; i < post_block_size; ++i, ++out_index, ++in_index)
        out_buf[out_index] = in_buf[in_index];
}""")

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"TransposeOp({dev})"

    __str__ = __repr__

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

        output = tensor_out.data().reshape(output_shape)
        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_out, shape_in = input_tensors[0]
        ndim = len(shape_in)

        if ndim <= 1:
            return [(tensor_out, shape_in)]

        if len(input_tensors) > 1:
            perm_vals = [d if d >= 0 else d + ndim for d in input_tensors[1][0].data().astype(int)]
            assert sorted(perm_vals) == list(range(ndim)), \
                f"Invalid permutation {perm_vals!r} with shape {shape_in!r}."
        else:
            perm_vals = list(reversed(range(ndim)))

        shape_out = [int(shape_in[i]) for i in perm_vals]
        total_size = np.prod(shape_out)
        leading_size = 1
        suffix = list(range(ndim))
        for i in range(ndim - 1):
            if suffix[0] != perm_vals[i]:
                tensor_in = tensor_out
                tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
                pre_block_size = shape_in[suffix[0]]
                j = 1
                while suffix[j] != perm_vals[i]:
                    pre_block_size *= shape_in[suffix[j]]
                    j += 1
                post_block_size = 1
                j += 1
                while j < len(suffix):
                    post_block_size *= shape_in[suffix[j]]
                    j += 1
                axis_dimension = shape_out[i]
                workgroup = (pre_block_size, leading_size, axis_dimension)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_out],
                    self.compiled_shader,
                    workgroup,
                    [pre_block_size, post_block_size, axis_dimension],
                    []
                ))
                updated_tensors.append(tensor_out)
            suffix.remove(perm_vals[i])
            leading_size *= shape_out[i]

        return [(tensor_out, shape_out)]