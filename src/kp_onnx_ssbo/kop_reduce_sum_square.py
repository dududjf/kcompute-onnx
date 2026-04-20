import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_1D, LOCAL_X_2D, LOCAL_Y_2D


class ReduceSumSquareOp:
    def __init__(self, manager: kp.Manager, keepdims=True, noop_with_empty_axes=False):
        self.manager = manager
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes

        # Element-wise square shader
        self.compiled_shader_square = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint total_size = params[0];
    if (gx >= total_size) return;

    out_tensor[gx] = in_tensor[gx] * in_tensor[gx];
}}
""")

        # Sum reduce shader
        self.compiled_shader_reduce = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint group_x = params[0];
    uint block_size = params[1];
    if (gx >= group_x || gy >= block_size) return;

    uint dimension = params[2];

    uint in_offset  = gx * dimension * block_size + gy;
    uint out_offset = gx * block_size + gy;

    float acc = 0.0;
    for (uint i = 0; i < dimension; ++i, in_offset += block_size) {{
        acc += in_tensor[in_offset];
    }}
    out_tensor[out_offset] = acc;
}}
""")

    def __repr__(self):
        return f"ReduceSumSquareOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp is not None:
                numpy_in = inp.reshape(-1).astype(np.float32) \
                    if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]] + updated_tensors))
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
        axes = input_tensors[1][0].data().astype(int) if len(input_tensors) > 1 else None

        tensor_in = input_tensors[0][0]
        shape_in = input_tensors[0][1]

        if self.noop_with_empty_axes and axes is None:
            size = int(np.prod(shape_in))
            tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
            updated_tensors.append(tensor_out)

            params = np.array([size], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            workgroup = ((size + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_out, param_in],
                self.compiled_shader_square,
                workgroup,
            ))
            return [(tensor_out, shape_in)]

        else:
            axis_present = [True if axes is None else False] * len(shape_in)
            if axes is not None:
                for axis in axes:
                    idx = axis if axis >= 0 else axis + len(shape_in)
                    axis_present[idx] = True

            # First: element-wise square
            size = int(np.prod(shape_in))
            tensor_sq = self.manager.tensor(np.zeros(size, dtype=np.float32))
            updated_tensors.append(tensor_sq)

            params_sq = np.array([size], dtype=np.uint32)
            param_sq_in = self.manager.tensor_t(params_sq, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_sq_in])).eval()

            workgroup_sq = ((size + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_sq, param_sq_in],
                self.compiled_shader_square,
                workgroup_sq,
            ))

            tensor_out = tensor_sq
            block_size = 1

            # Then: reduce sum along axes (right to left)
            for i in reversed(range(len(shape_in))):
                if axis_present[i]:
                    group_x = int(np.prod(shape_in[:i])) if i > 0 else 1
                    numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                    prev_in = tensor_out
                    tensor_out = self.manager.tensor(numpy_out)
                    updated_tensors.append(tensor_out)

                    params = np.array([group_x, block_size, shape_in[i]], dtype=np.uint32)
                    param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                    self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                    workgroup = (
                        (group_x + LOCAL_X_2D - 1) // LOCAL_X_2D,
                        (block_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                        1,
                    )

                    updated_algorithms.append(self.manager.algorithm(
                        [prev_in, tensor_out, param_in],
                        self.compiled_shader_reduce,
                        workgroup,
                    ))
                else:
                    block_size *= int(shape_in[i])

            if self.keepdims:
                shape_out = [1 if axis_present[i] else shape_in[i] for i in range(len(shape_in))]
            else:
                shape_out = [shape_in[i] for i in range(len(shape_in)) if not axis_present[i]]

            return [(tensor_out, shape_out)]
