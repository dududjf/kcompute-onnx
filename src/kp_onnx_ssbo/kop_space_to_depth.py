import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class SpaceToDepthOp:
    def __init__(self, manager: kp.Manager, blocksize=None):
        self.manager = manager
        self.blocksize = blocksize
        self.compiled_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint leading_idx = gl_GlobalInvocationID.x;
    uint pre_idx = gl_GlobalInvocationID.y;
    uint axis_idx = gl_GlobalInvocationID.z;

    uint leading_size = params[0];
    uint pre_block_size = params[1];
    uint axis_dimension = params[2];
    uint post_block_size = params[3];
    if (leading_idx >= leading_size || pre_idx >= pre_block_size || axis_idx >= axis_dimension) return;

    uint stride_y = axis_dimension * post_block_size;
    uint stride_x = pre_block_size * stride_y;
    uint in_index = leading_idx * stride_x + pre_idx * stride_y + axis_idx * post_block_size;
    uint out_index = leading_idx * stride_x + axis_idx * pre_block_size * post_block_size + pre_idx * post_block_size;

    for (uint i = 0; i < post_block_size; ++i, ++out_index, ++in_index)
        out_tensor[out_index] = in_tensor[in_index];
}}
""")

    def __repr__(self):
        return f"SpaceToDepthOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

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

        tmpshape = [batch_size, input_channels, input_height // blocksize, blocksize,
                    input_width // blocksize, blocksize]

        perm_vals = [0, 3, 5, 1, 2, 4]
        shape_out = [tmpshape[i] for i in perm_vals]
        total_size = int(np.prod(shape_out))
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

                params = np.array([leading_size, pre_block_size, axis_dimension,
                                   post_block_size], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                workgroup = (
                    (leading_size + LOCAL_X_3D - 1) // LOCAL_X_3D,
                    (pre_block_size + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                    (axis_dimension + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
                )

                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_out, param_in],
                    self.compiled_shader,
                    workgroup,
                ))
                updated_tensors.append(tensor_out)
            suffix.remove(perm_vals[i])
            leading_size *= shape_out[i]

        shape_out = [batch_size, input_channels * blocksize * blocksize,
                     input_height // blocksize, input_width // blocksize]
        return [(tensor_out, shape_out)]
