import numpy as np
import kp
from .shader_utils import compile_source


class SpaceToDepthOp:
    def __init__(self, manager: kp.Manager, blocksize=None):
        self.manager = manager
        self.blocksize = blocksize
        self.compiled_shader = compile_source('''
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
}''')

    def __repr__(self):
        return f"SpaceToDepthOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]]))
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
        assert len(shape_in) == 4, "SpaceToDepthOp requires 4D input tensor"

        blocksize = self.blocksize
        batch_size, input_channels, input_height, input_width = shape_in

        tmpshape = [batch_size, input_channels, input_height // blocksize, blocksize, input_width // blocksize, blocksize]

        perm_vals = [0, 3, 5, 1, 2, 4]
        shape_out = [tmpshape[i] for i in perm_vals]
        total_size = np.prod(shape_out)
        leading_size = 1
        suffix = list(range(len(tmpshape)))
        for i in range(len(tmpshape) - 1):
            if suffix[0] != perm_vals[i]:
                tensor_in = tensor_out
                tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
                pre_block_size = tmpshape[suffix[0]]
                j = 1
                while suffix[j] != perm_vals[i]:
                    pre_block_size *= tmpshape[suffix[j]]
                    j += 1
                post_block_size = 1
                j += 1
                while j < len(suffix):
                    post_block_size *= tmpshape[suffix[j]]
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
        shape_out = [batch_size, input_channels * blocksize * blocksize, input_height // blocksize, input_width // blocksize]
        return [(tensor_out, shape_out)]