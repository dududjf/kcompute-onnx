import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_1D


class ErfOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source(f"""
#version 450
layout(local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint total_size = params[0];

    if (gx >= total_size) return;

    float a1 =  0.254829592;
    float a2 = -0.284496736;
    float a3 =  1.421413741;
    float a4 = -1.453152027;
    float a5 =  1.061405429;
    float p  =  0.3275911;

    float x = in_tensor[gx];
    float s = sign(x);
    float x_abs = abs(x);
    float t = 1.0 / (1.0 + p * x_abs);
    float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x_abs * x_abs);
    out_tensor[gx] = s * y;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ErfOp({device_name})"

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
        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]
        size = int(np.prod(tensor_shape))
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params = np.array([size], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        workgroup = ((size + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out, param_in],
            self.shader,
            workgroup,
        ))

        return [(tensor_out, tensor_shape)]
