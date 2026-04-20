import numpy as np
import kp
from .shader_utils import compile_source


class ShrinkOp:
    def __init__(self, manager: kp.Manager, bias=0.0, lambd=0.5):
        self.manager = manager
        self.bias = bias
        self.lambd = lambd
        self.shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf     { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf    { float out_data[]; };

layout(constant_id=0) const float Bias = 0;
layout(constant_id=1) const float Lambda = 0;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float x = in_data[i];
    float y = (x < -Lambda) ? (x + Bias) : (x > Lambda) ? (x - Bias) : 0.0;
    out_data[i] = y;
}
""")

    def __repr__(self):
        return f"ShrinkOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

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
        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]
        size = np.prod(tensor_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))

        updated_tensors.append(tensor_out)
        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out],
                                                         self.shader, spec_consts=[self.bias, self.lambd]))
        return [(tensor_out, tensor_shape)]
