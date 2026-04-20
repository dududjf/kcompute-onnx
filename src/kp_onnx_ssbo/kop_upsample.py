import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class UpsampleOp:
    def __init__(self, manager: kp.Manager, mode=None):
        self.manager = manager
        self.mode = mode
        self.compiled_shader_nearest = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint group_count = params[0];
    uint scale = params[1];
    if (gx >= group_count || gy >= scale) return;

    uint in_block_size = params[2];
    uint out_block_size = params[3];

    uint in_offset = gx * in_block_size;
    uint out_offset = gx * out_block_size + gy * in_block_size;

    for (uint i = 0; i < in_block_size; i++, in_offset++, out_offset++)
        out_tensor[out_offset] = in_tensor[in_offset];
}}
""")

    def __repr__(self):
        return f"UpsampleOp({self.manager.get_device_properties()['device_name']})"

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
        assert len(input_tensors) == 2, "UpsampleOp needs input tensor and repeats"

        data_tensor = input_tensors[0][0]
        data_shape = input_tensors[0][1]
        scales = input_tensors[1][0].data().astype(int)
        assert self.mode == "nearest", f"Not implemented for mode={self.mode!r} and scale={scales!r}."

        tensor_out = data_tensor
        block_size = 1

        for i in reversed(range(len(data_shape))):
            dimension = data_shape[i]
            if scales[i] > 1:
                data_tensor = tensor_out
                in_block_size = block_size
                out_block_size = int(in_block_size * scales[i])
                group_count = int(np.prod(data_shape[:i])) * dimension if i > 0 else dimension

                out_array = np.zeros(group_count * out_block_size, dtype=np.float32)
                tensor_out = self.manager.tensor(out_array)

                params = np.array([group_count, scales[i], in_block_size, out_block_size], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                workgroup = (
                    (group_count + LOCAL_X_2D - 1) // LOCAL_X_2D,
                    (scales[i] + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                    1,
                )

                updated_algorithms.append(self.manager.algorithm(
                    [data_tensor, tensor_out, param_in],
                    self.compiled_shader_nearest,
                    workgroup,
                ))
                updated_tensors.append(tensor_out)
                block_size = out_block_size * dimension
            else:
                block_size *= dimension

        return [(tensor_out, list(np.array(data_shape) * scales))]
