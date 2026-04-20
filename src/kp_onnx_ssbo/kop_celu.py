import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D


class CeluOp:
    """
    onnx::Celu 的 Kompute 实现（逐元素一元 | float32）
    CELU(x) = x (x>=0) ; alpha * (exp(x/alpha) - 1) (x<0)
    """

    def __init__(self, manager: kp.Manager, alpha=1.0):
        self.alpha = alpha
        self.manager = manager
        self.compiled_shader = compile_source(f'''
#version 450

layout (local_size_x = {LOCAL_X_1D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf {{ float in_buf[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{
    uint bound_x;
    float alpha;
}};

void main()
{{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= bound_x) return;
    float x = in_buf[idx];
    // CELU(x) = x (x>=0) ; alpha * (exp(x/alpha) - 1) (x<0)
    out_buf[idx] = (x >= 0.0) ? x : (alpha * (exp(x / alpha) - 1.0));
}}''')

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"CeluOp({dev})"

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
        tensor_in, shape = input_tensors[0]
        size = int(np.prod(shape)) if len(shape) > 0 else 1

        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        # 创建参数张量并立即同步到GPU
        param_in = self.manager.tensor_t(np.array([size, self.alpha], dtype=np.float32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        workgroup = ((size + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out, param_in],
            self.compiled_shader,
            workgroup
        ))
        return [(tensor_out, shape)]

