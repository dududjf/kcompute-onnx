import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class MatMulOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.sizes = None
        self.workgroup = None
        self.shader = None
        self.shader_code1 = f"""
#version 450

layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout (std430, set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly buffer UIParams {{ uint params[]; }};

void main()
{{
    uint size_m = params[0];
    uint size_n = params[1];
    uint size_k = params[2];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    if(row >= size_m || col >= size_n) return;
    float acc = 0.0;
    uint start_1 = row * size_k;
    for(uint i = 0, start_2 = col; i < size_k; i++, start_2 += size_n)
        acc += in_tensor_1[start_1 + i] * in_tensor_2[start_2];
    out_tensor[(row * size_n) + col] = acc;
}}
"""
        self.shader_code2 = f"""
#version 450

layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly buffer UIParams {{ uint params[]; }};

void main()
{{
    uint size_m = params[0];
    uint size_n = params[1];
    uint size_b = params[2];
    uint size_k = params[3];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint batch = gl_GlobalInvocationID.z;
    if(row >= size_m || col >= size_n || batch >= size_b) return;
    float acc = 0.0;
    uint start_1 = (batch * size_m * size_k) + (row * size_k);
    uint start_2 = (batch * size_k * size_n) + col;
    for(uint i = 0; i < size_k; i++, start_2 += size_n)
        acc += in_tensor_1[start_1 + i] * in_tensor_2[start_2];
    out_tensor[(batch * size_m * size_n) + (row * size_n) + col] = acc;
}}
"""

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
            rows = int(np.prod(shape_1[:-1]))
            cols = shape_1[-1]
            nrows = shape_2[0]
            ncols = shape_2[1]
            assert cols == nrows, f"MatMulOp requires #columns {cols} of the 1st and #rows {nrows} of the 2nd to equal"
            tensor_out = self.manager.tensor(np.zeros(rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols]:
                self.sizes = [rows, cols, ncols]
                self.shader = compile_source(self.shader_code1)
                self.workgroup = (
                    (rows + LOCAL_X_2D - 1) // LOCAL_X_2D, (ncols + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1)

            updated_tensors.append(tensor_out)
            params = np.array([rows, ncols, cols], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out, param_in],
                                                             self.shader, self.workgroup))

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
            blocks = int(np.prod(shape_1[:-2]))
            tensor_out = self.manager.tensor(np.zeros(blocks * rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols, blocks]:
                self.sizes = [rows, cols, ncols, blocks]
                self.shader = compile_source(self.shader_code2)
                self.workgroup = (
                    (rows + LOCAL_X_3D - 1) // LOCAL_X_3D, (ncols + LOCAL_Y_3D - 1) // LOCAL_Y_3D, (blocks + LOCAL_Z_3D - 1) // LOCAL_Z_3D)

            updated_tensors.append(tensor_out)
            params = np.array([rows, ncols, blocks, cols], dtype=np.uint32)
            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out, param_in],
                                                             self.shader, self.workgroup))

            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]
