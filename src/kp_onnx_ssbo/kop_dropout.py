import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D

DEFAULT_RATIO = 0.5
DEFAULT_TRAINING_MODE = False


class DropoutOp:
    def __init__(self, manager: kp.Manager, seed=None):
        self.manager = manager
        self.seed = seed
        self.compiled_shader = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_1D}) in;
layout (std430, set=0, binding=0) readonly  buffer InBuf    {{ float in_tensor[];  }};
layout (std430, set=0, binding=1) readonly  buffer MaskBuf  {{ float mask_tensor[]; }};
layout (std430, set=0, binding=2) writeonly buffer OutBuf   {{ float out_tensor[];  }};
layout (std430, set=0, binding=3) readonly  buffer ScaleBuf {{ float scale_tensor[]; }};
layout (std430, set=0, binding=4) readonly  buffer UIParams {{ uint bound_x; }};

void main() 
{{
    uint gx = gl_GlobalInvocationID.x;
    if (gx >= bound_x) return;

    float scale = scale_tensor[0];
    out_tensor[gx] = in_tensor[gx] * mask_tensor[gx] * scale;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"DropoutOp({device_name})"

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

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
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

        ratio = float(input_tensors[1][0].data()) if len(input_tensors) >= 2 else DEFAULT_RATIO
        training_mode = bool(input_tensors[2][0].data()) if len(input_tensors) >= 3 else DEFAULT_TRAINING_MODE

        if not training_mode or ratio == 0.0:
            return [(tensor_in, in_shape)]

        rnd = np.random.RandomState(None if self.seed is None else int(self.seed))
        keep_prob = 1.0 - float(ratio)
        mask_bool = rnd.uniform(0.0, 1.0, in_shape) < keep_prob
        scale = 1.0 / (1.0 - ratio)

        size = int(np.prod(in_shape))
        tensor_mask = self.manager.tensor(mask_bool.reshape(-1).astype(np.float32))
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        tensor_scale = self.manager.tensor(np.array([scale], dtype=np.float32))

        params = np.array([size], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([tensor_scale, param_in])).eval()

        updated_tensors.extend([tensor_mask, tensor_out])

        group_x = (size + LOCAL_X_1D - 1) // LOCAL_X_1D
        workgroup = (group_x, 1, 1)

        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_mask, tensor_out, tensor_scale, param_in],
                self.compiled_shader,
                workgroup
            )
        )
        return [(tensor_out, in_shape)]
