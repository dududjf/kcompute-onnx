import kp
import numpy as np
from .shader_utils import compile_source


class SeluOp:

    def __init__(self, manager: kp.Manager, alpha=float(1.6732631921768188), gamma=float(1.0507009873554805)):
        self.alpha = alpha
        self.gamma = gamma
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float alpha = 0;
layout (constant_id = 1) const float gamma = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float x = in_buf[idx];
    // SELU(x) = gamma * ( x (x>=0) ; alpha*(exp(x)-1) (x<0) )
    float y = x >= 0.0 ? x : alpha * (exp(x) - 1.0);
    out_buf[idx] = gamma * y;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"SeluOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

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
        tensor_in, tensor_shape = input_tensors[0]
        size = np.prod(tensor_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (size, 1, 1)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            [self.alpha, self.gamma],
            []
        ))
        return [(tensor_out, tensor_shape)]
