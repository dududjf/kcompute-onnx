import kp
import numpy as np
from .shader_utils import compile_source


class NegOp:
    """
    onnx::Neg 的 Kompute 实现（逐元素一元 | float32）
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    out_buf[idx] = -in_buf[idx];
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"NegOp({dev})"

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

        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out],
                self.shader,
                (size, 1, 1),
                [],
                []
            )
        )
        return [(tensor_out, shape)]
