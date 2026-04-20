import numpy as np
import kp
from .shader_utils import compile_source


class SliceOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly  buffer in_buf     { float in_tensor[];     };
layout(binding = 1) writeonly buffer out_buf    { float out_tensor[];    };

layout(constant_id = 0) const float elements_per_slice_f = 0;  // 每个切片的元素数量
layout(constant_id = 1) const float start_f = 0;          // 切片的起始位置
layout(constant_id = 2) const float size_f = 0;           // 切片的大小
layout(constant_id = 3) const float axis_size_f = 0;      // 切片轴的大小
layout(constant_id = 4) const float step_f = 0;           // 步长

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint elements_per_slice = uint(elements_per_slice_f);
    uint start = uint(start_f);
    uint size = uint(size_f);
    uint axis_size = uint(axis_size_f);
    uint step = uint(step_f);

    uint in_offset = (gx * axis_size + start + gy * step) * elements_per_slice + gz;
    uint out_offset = (gx * size + gy) * elements_per_slice + gz;

    out_tensor[out_offset] = in_tensor[in_offset];
}
""")

    def __repr__(self):
        return f"SliceOp({self.manager.get_device_properties()['device_name']})"

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
                empty_array = np.array([], dtype=np.float32).reshape(shape)
                outputs.append(empty_array)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        starts = input_tensors[1][0].data().astype(int)
        ends = input_tensors[2][0].data().astype(int)
        axes = input_tensors[3][0].data().astype(int) if len(input_tensors) > 3 else None
        steps = input_tensors[4][0].data().astype(int) if len(input_tensors) > 4 else None

        if axes is None:
            axes = list(range(len(starts)))
        else:
            axes = [a + len(shape_in) if a < 0 else a for a in axes]

        for i, axis in enumerate(axes):
            if starts[i] < 0:
                starts[i] = starts[i] + shape_in[i]
            if ends[i] < 0:
                ends[i] = ends[i] + shape_in[i]

        if steps is None:
            steps = [1] * len(axes)

        tensor_out = tensor_in
        current_shape = shape_in

        for i in range(len(axes)):
            axis = axes[i]

            start = starts[i]
            end = ends[i]
            step = steps[i]

            size = (end - start + step - 1) // step

            shape_out = current_shape.copy()
            shape_out[axis] = size

            pre_elements = int(np.prod(shape_out[:axis])) if axis > 0 else 1
            elements_per_slice = int(np.prod(shape_out[axis + 1:])) if axis + 1 < len(shape_out) else 1

            total_elements = pre_elements * size * elements_per_slice

            if total_elements == 0:
                return [(None, shape_out)]

            tensor_in = tensor_out
            # 创建输出张量
            tensor_out = self.manager.tensor(np.zeros(total_elements, dtype=np.float32))
            updated_tensors.append(tensor_out)

            workgroup = (pre_elements, size, elements_per_slice)
            updated_algorithms.append(
                self.manager.algorithm(
                    [tensor_in, tensor_out],
                    self.compiled_shader,
                    workgroup,
                    [elements_per_slice, start, size, current_shape[axis], step],
                    []
                )
            )

            current_shape = shape_out

        return [(tensor_out, current_shape)]
