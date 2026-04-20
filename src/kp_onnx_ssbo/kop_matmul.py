import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class MatMulOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader1 = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf1 {{ float in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer InBuf2 {{ float in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main()
{{
    uint size_m = params[0], size_k = params[1], size_n = params[2];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    if(row >= size_m || col >= size_n) return;
    float acc = 0.0;
    for(uint i = 0, start_1 = row * size_k, start_2 = col; i < size_k; ++i, ++start_1, start_2 += size_n)
        acc += in_tensor_1[start_1] * in_tensor_2[start_2];
    out_tensor[row * size_n + col] = acc;
}}
""")
        self.compiled_shader2 = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf1 {{ float in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer InBuf2 {{ float in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main()
{{
    uint size_m = params[0], size_k = params[1], size_n = params[2], size_b = params[3];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint batch = gl_GlobalInvocationID.z;
    if(row >= size_m || col >= size_n || batch >= size_b) return;
    float acc = 0.0;
    uint start_1 = batch * size_m * size_k + row * size_k;
    uint start_2 = batch * size_k * size_n + col;
    for(uint i = 0; i < size_k; ++i, ++start_1, start_2 += size_n)
        acc += in_tensor_1[start_1] * in_tensor_2[start_2];
    out_tensor[batch * size_m * size_n + row * size_n + col] = acc;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        assert len(inputs) == 2, "MatMulOp requires 2 inputs"

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
        assert len(input_tensors) == 2, "MatMulOp requires 2 inputs"
        tensor_in_1 = input_tensors[0][0]
        tensor_in_2 = input_tensors[1][0]
        shape_1 = input_tensors[0][1]
        shape_2 = input_tensors[1][1]

        if len(shape_1) >= 2 and len(shape_2) == 2:
            rows = np.prod(shape_1[:-1])
            cols = shape_1[-1]
            nrows = shape_2[0]
            ncols = shape_2[1]
            assert cols == nrows, f"MatMulOp requires #columns {cols} of the 1st and #rows {nrows} of the 2nd to equal"

            params = [rows, cols, ncols]
            param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            tensor_out = self.manager.tensor(np.zeros(rows * ncols, dtype=np.float32))
            group_x = (rows + LOCAL_X_2D - 1) // LOCAL_X_2D
            group_y = (ncols + LOCAL_Y_2D - 1) // LOCAL_Y_2D

            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out, param_in],
                                                             self.compiled_shader1,
                                                             (group_x, group_y, 1)))

            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]

        else:
            assert 2 < len(shape_1) == len(shape_2) and shape_1[:-2] == shape_2[:-2], \
                f"MatMulOp requires the prefix dimensions {shape_1[:-2]} and {shape_2[:-2]} to equal"
            rows = shape_1[-2]
            cols = shape_1[-1]
            nrows = shape_2[-2]
            ncols = shape_2[-1]
            assert cols == nrows, f"MatMulOp requires #columns {cols} of the 1st and #rows {nrows} of the 2nd to equal"

            blocks = np.prod(shape_1[:-2])
            params = [rows, cols, ncols, blocks]
            param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            tensor_out = self.manager.tensor(np.zeros(blocks * rows * ncols, dtype=np.float32))
            group_x = (rows + LOCAL_X_3D - 1) // LOCAL_X_3D
            group_y = (ncols + LOCAL_Y_3D - 1) // LOCAL_Y_3D
            group_z = (blocks + LOCAL_Z_3D - 1) // LOCAL_Z_3D

            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out, param_in],
                                                             self.compiled_shader2,
                                                             (group_x, group_y, group_z)))

            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]
