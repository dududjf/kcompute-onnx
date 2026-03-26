import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class ScalerOp:
    def __init__(self, manager: kp.Manager, offset=None, scale=None):
        self.manager = manager
        self.offset = offset
        self.scale = scale

        self.compiled_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf     {{ float in_tensor[];     }};
layout (std430, set = 0, binding = 1) readonly  buffer OffsetBuf {{ float offset_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer ScaleBuf  {{ float scale_buf[];  }};
layout (std430, set = 0, binding = 3) writeonly buffer OutBuf    {{ float out_tensor[];    }};
layout (std430, set = 0, binding = 4) readonly  buffer UIParams  {{ uint params[]; }};

void main() {{
    uint leading_idx = gl_GlobalInvocationID.x;
    uint trailing_idx = gl_GlobalInvocationID.y;

    uint leading = params[0];
    uint trailing = params[1];
    if (leading_idx >= leading || trailing_idx >= trailing) return;

    uint idx = leading_idx * trailing + trailing_idx;
    out_tensor[idx] = (in_tensor[idx] - offset_buf[trailing_idx]) * scale_buf[trailing_idx];
}}
""")

    def __repr__(self):
        return f"ScalerOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

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

        tensor_offset = self.manager.tensor(np.array(offset, dtype=np.float32))
        tensor_scale = self.manager.tensor(np.array(scale, dtype=np.float32))
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.extend([tensor_offset, tensor_scale, tensor_out])

        params = np.array([leading, trailing], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        workgroup = (
            (leading + LOCAL_X_2D - 1) // LOCAL_X_2D,
            (trailing + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
            1,
        )

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_offset, tensor_scale, tensor_out, param_in],
            self.compiled_shader,
            workgroup,
        ))
        return [(tensor_out, list(shape_in))]
