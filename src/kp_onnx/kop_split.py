import numpy as np
import kp
from .shader_utils import compile_source


class SplitOp:
    def __init__(self, manager: kp.Manager, axis=0, num_outputs=None):
        self.manager = manager
        self.axis = axis
        self.num_outputs = num_outputs
        self.compiled_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly  buffer in_buf     { float in_tensor[];     };
layout(binding = 1) writeonly buffer out_buf    { float out_tensor[];    };

layout(constant_id = 0) const float pre_elements_f = 0; // 每个切片之前的元素数量
layout(constant_id = 1) const float elements_per_slice_f = 0;   // 每个切片的元素数量
layout(constant_id = 2) const float offset_f = 0;       // 切片的起始偏移量
layout(constant_id = 3) const float size_f = 0;         // 输出切片的大小
layout(constant_id = 4) const float axis_size_f = 0;    // 分割轴的大小

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    uint pre_elements = uint(pre_elements_f);
    uint elements_per_slice = uint(elements_per_slice_f);
    uint offset = uint(offset_f);
    uint size = uint(size_f);
    uint axis_size = uint(axis_size_f);
    
    uint in_base_offset = gx * axis_size * elements_per_slice + offset * elements_per_slice;
    uint out_offset = gx * size * elements_per_slice + gy;
    
    uint in_offset = in_base_offset + gy;
    
    out_tensor[out_offset] = in_tensor[in_offset];
}
""")

    def __repr__(self):
        return f"SplitOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensors_and_shapes = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))

        # 只同步非空张量
        non_empty_tensors = [tensor for tensor, _ in output_tensors_and_shapes if tensor is not None]
        if non_empty_tensors:
            seq.record(kp.OpTensorSyncLocal(non_empty_tensors))
        seq.eval()

        outputs = []
        for tensor, shape in output_tensors_and_shapes:
            if tensor is not None:
                output = tensor.data().reshape(shape)
                outputs.append(output)
            else:
                # 创建空张量
                empty_array = np.array([], dtype=np.float32).reshape(shape)
                outputs.append(empty_array)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        split = input_tensors[1][0].data().astype(int) if len(input_tensors) > 1 else None
        axis = self.axis

        if axis < 0:
            axis += len(shape_in)

        axis_size = shape_in[axis]
        output_tensors_and_shapes = []

        # 计算分割大小
        if split is None:
            if shape_in[axis] % self.num_outputs == 0:
                div = shape_in[axis] // self.num_outputs
                split = [div] * self.num_outputs
            else:
                div = shape_in[axis] // self.num_outputs + 1
                split = [div] * self.num_outputs
                split[-1] += shape_in[axis] - sum(split)

        assert sum(split) == axis_size, \
            f"Sum of split values ({sum(split)}) must equal the size of axis {axis} ({axis_size})"

        pre_elements = np.prod(shape_in[:axis]) if axis > 0 else 1
        elements_per_slice = np.prod(shape_in[axis + 1:]) if axis < len(shape_in) - 1 else 1

        offset = 0
        for size in split:
            shape_out = shape_in[:axis] + [size] + shape_in[axis + 1:]

            if size == 0:
                # 对于大小为0或负数的情况，不创建实际张量，只记录形状
                output_tensors_and_shapes.append((None, shape_out))
                continue

            total_elements = int(pre_elements * size * elements_per_slice)

            tensor_out = self.manager.tensor(np.zeros(total_elements, dtype=np.float32))
            updated_tensors.append(tensor_out)

            workgroup = (pre_elements, size * elements_per_slice, 1)

            updated_algorithms.append(
                self.manager.algorithm(
                    [tensor_in, tensor_out],
                    self.compiled_shader,
                    workgroup,
                    [pre_elements, elements_per_slice, offset, size, axis_size],
                    []
                )
            )

            output_tensors_and_shapes.append((tensor_out, shape_out))
            offset += size

        return output_tensors_and_shapes
