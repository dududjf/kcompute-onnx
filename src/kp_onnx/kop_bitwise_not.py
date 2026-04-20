import kp
import numpy as np
from pyshader import python2shader, ivec2, Array, i32
from .shader_utils import compile_source


@python2shader
def compute_shader_bitwise_not_int(index=("input", "GlobalInvocationId", ivec2),
                                   in_data=("buffer", 0, Array(i32)),
                                   out_data=("buffer", 1, Array(i32))):
    i = index.x
    out_data[i] = -in_data[i] - 1

_bitwise_not_code_int = compute_shader_bitwise_not_int.to_spirv()


class BitwiseNotOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

        self.shader_float = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf  { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf { float out_data[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    int bits = floatBitsToInt(in_data[i]);
    bits = ~bits;
    out_data[i] = intBitsToFloat(bits);
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp.dtype == np.float32:
                numpy_in = inp.reshape(-1)
                tensor = self.manager.tensor(numpy_in)
            elif inp.dtype == np.int32:
                numpy_in = inp.reshape(-1)
                tensor = self.manager.tensor_t(numpy_in, tensor_type=kp.TensorTypes.device)
            else:
                raise TypeError(f"Unsupported dtype {inp.dtype}. Only float32 and int32 supported.")
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

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]],updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]
        size = np.prod(tensor_shape)
        dtype = tensor_in.data().dtype

        if dtype == np.float32:
            tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out], self.shader_float))

        elif dtype == np.int32:
            tensor_out = self.manager.tensor_t(np.zeros(size, dtype=np.int32), tensor_type=kp.TensorTypes.device)
            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out], _bitwise_not_code_int))

        else:
            raise TypeError(f"Unsupported tensor dtype {dtype}. Only float32 and int32 supported.")

        return [(tensor_out, tensor_shape)]
