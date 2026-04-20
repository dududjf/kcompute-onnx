import numpy as np
import kp
from .shader_utils import compile_source


class TileOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) buffer buf_in_tensor { float in_tensor[]; };
layout (binding = 1) buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float in_block_size_f = 0;
layout (constant_id = 1) const float out_block_size_f = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint in_block_size = uint(in_block_size_f);
    uint out_block_size = uint(out_block_size_f);

    uint in_offset = gx * in_block_size;
    uint out_offset = gx * out_block_size + gy * in_block_size;

    for (uint i = 0; i < in_block_size; i++, in_offset++, out_offset++)
        out_tensor[out_offset] = in_tensor[in_offset];
}
""")

    def __repr__(self):
        return f"TileOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        assert len(inputs) == 2, "TileOp needs two inputs tensor and repeats"

        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        # 执行GPU操作序列
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
        assert len(input_tensors) == 2, "TileOp needs two inputs tensor and repeats"

        data_tensor = input_tensors[0][0]
        data_shape = input_tensors[0][1]
        repeats = input_tensors[1][0].data().astype(int)

        tensor_out = data_tensor
        block_size = 1

        for i in reversed(range(len(data_shape))):
            dimension = data_shape[i]
            if repeats[i] > 1:
                data_tensor = tensor_out
                in_block_size = block_size * dimension
                out_block_size = in_block_size * repeats[i]
                group_count = np.prod(data_shape[:i]) if i > 0 else 1
                out_array = np.zeros(group_count * out_block_size, dtype=np.float32)
                tensor_out = self.manager.tensor(out_array)
                workgroup = (group_count, repeats[i], 1)
                updated_algorithms.append(self.manager.algorithm([data_tensor, tensor_out],
                                                                 self.compiled_shader,
                                                                 workgroup,
                                                                 [in_block_size, out_block_size],
                                                                 []))
                updated_tensors.append(tensor_out)
                block_size = out_block_size
            else:
                block_size *= dimension

        return [(tensor_out, data_shape * repeats)]
