import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_1D


class ClipOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader_min = compile_source(f"""
#version 450

layout(local_size_x = {LOCAL_X_1D}) in;
layout(std430, set=0, binding=0) readonly  buffer InBuf    {{ float in_tensor[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf   {{ float out_tensor[]; }};
layout(std430, set=0, binding=2) readonly  buffer MinBuf   {{ float min_tensor[];  }};
layout(std430, set=0, binding=3) readonly  buffer UIParams {{ uint bound_x; }};

void main() 
{{
    uint gx = gl_GlobalInvocationID.x;
    if (gx >= bound_x) return;

    float v = in_tensor[gx];
    float min_value = min_tensor[0];
    out_tensor[gx] = (v < min_value) ? min_value : v;
}}
""")

        self.shader_minmax = compile_source(f"""
#version 450

layout(local_size_x = {LOCAL_X_1D}) in;
layout(std430, set=0, binding=0) readonly  buffer InBuf    {{ float in_tensor[];  }};
layout(std430, set=0, binding=1) writeonly buffer OutBuf   {{ float out_tensor[]; }};
layout(std430, set=0, binding=2) readonly  buffer MinBuf   {{ float min_tensor[];  }};
layout(std430, set=0, binding=3) readonly  buffer MaxBuf   {{ float max_tensor[];  }};
layout(std430, set=0, binding=4) readonly  buffer UIParams {{ uint bound_x; }};

void main() 
{{
    uint gx = gl_GlobalInvocationID.x;
    if (gx >= bound_x) return;

    float v = in_tensor[gx];
    float min_value = min_tensor[0];
    float max_value = max_tensor[0];
    v = (v < min_value) ? min_value : v;
    v = (v > max_value) ? max_value : v;
    out_tensor[gx] = v;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ClipOp({device_name})"

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
        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]
        size = int(np.prod(tensor_shape))
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))

        tensor_min = input_tensors[1][0] if len(input_tensors) > 1 else None
        tensor_max = input_tensors[2][0] if len(input_tensors) > 2 else None

        if tensor_min is None and tensor_max is None:
            return [(tensor_in, tensor_shape)]

        group_x = (size + LOCAL_X_1D - 1) // LOCAL_X_1D
        workgroup = (group_x, 1, 1)

        params = np.array([size], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        updated_tensors.append(tensor_out)

        if tensor_min is not None and tensor_max is None:
            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_out, tensor_min, param_in],
                self.shader_min,
                workgroup
            ))
            return [(tensor_out, tensor_shape)]

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out, tensor_min, tensor_max, param_in],
            self.shader_minmax,
            workgroup
        ))
        return [(tensor_out, tensor_shape)]
