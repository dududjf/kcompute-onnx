import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class SplitToSequenceOp:
    def __init__(self, manager: kp.Manager, axis=0, keepdims=1):
        self.manager = manager
        self.axis = axis
        self.keepdims = keepdims
        self.compiled_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint pre_elements = params[0];
    uint max_y = params[1];
    if (gx >= pre_elements || gy >= max_y) return;

    uint elements_per_slice = params[2];
    uint offset = params[3];
    uint size = params[4];
    uint axis_size = params[5];

    uint in_base_offset = gx * axis_size * elements_per_slice + offset * elements_per_slice;
    uint out_offset = gx * size * elements_per_slice + gy;
    uint in_offset = in_base_offset + gy;

    out_tensor[out_offset] = in_tensor[in_offset];
}}
""")

    def __repr__(self):
        return f"SplitToSequenceOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

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
        split = input_tensors[1][0].data().astype(int) if len(input_tensors) > 1 else None
        axis = self.axis

        axis += len(shape_in) if axis < 0 else 0

        axis_size = shape_in[axis]
        output_tensors_and_shapes = []

        if split is None:
            split_length = [1] * shape_in[axis]
        elif split.size == 1:
            dim = shape_in[axis]
            length = int(split)
            n = dim // length
            split_length = [length] * n
            left = dim - length * n
            if left > 0:
                split_length.append(left)
        else:
            split_length = list(split)

        assert sum(split_length) == axis_size, \
            f"Sum of split values ({sum(split_length)}) must equal the size of axis {axis} ({axis_size})"

        pre_elements = int(np.prod(shape_in[:axis])) if axis > 0 else 1
        elements_per_slice = int(np.prod(shape_in[axis + 1:])) if axis < len(shape_in) - 1 else 1

        offset = 0
        for size in split_length:
            shape_out = shape_in[:axis] + [size] + shape_in[axis + 1:]

            if split is None and not self.keepdims:
                shape_out = shape_out[:axis] + shape_out[axis + 1:]

            if size == 0:
                output_tensors_and_shapes.append((None, shape_out))
                continue

            total_elements = int(pre_elements * size * elements_per_slice)

            tensor_out = self.manager.tensor(np.zeros(total_elements, dtype=np.float32))
            updated_tensors.append(tensor_out)

            workgroup_y = size * elements_per_slice
            params = np.array([pre_elements, workgroup_y,
                               elements_per_slice, offset, size, axis_size], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            workgroup = (
                (pre_elements + LOCAL_X_2D - 1) // LOCAL_X_2D,
                (workgroup_y + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                1,
            )

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_out, param_in],
                self.compiled_shader,
                workgroup,
            ))

            output_tensors_and_shapes.append((tensor_out, shape_out))
            offset += size

        return output_tensors_and_shapes
