import kp
import numpy as np
from .shader_utils import compile_source


class MatMulOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

        props = manager.get_device_properties()
        max_workgroup_invocation = props['max_work_group_invocations']
        max_workgroup_size = props['max_work_group_size']
        local_size_x = 1
        local_size_y = 1
        while 2 * local_size_x * local_size_y <= max_workgroup_invocation:
            if 2 * local_size_x <= max_workgroup_size[0]:
                local_size_x *= 2
            if 2 * local_size_y <= max_workgroup_size[1]:
                local_size_y *= 2
            elif 2 * local_size_x > max_workgroup_size[0]:  # stop if neither x nor y can be double
                break

        self.local_size_x = local_size_x
        self.local_size_y = local_size_y
        self.sizes = None
        self.workgroup = None
        self.shader = None
        self.shader_code1 = """
#version 450

layout (local_size_x = {local_size_x}, local_size_y = {local_size_y}) in;
layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
layout (constant_id = 0) const float size_m_f = 0;
layout (constant_id = 1) const float size_k_f = 0;
layout (constant_id = 2) const float size_n_f = 0;

void main()
{{
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint size_m = uint(size_m_f);
    uint size_k = uint(size_k_f);
    uint size_n = uint(size_n_f);
    if(row >= size_m || col >= size_n) return;
    float acc = 0.0;
    for(uint i = 0, start_1 = row * size_k, start_2 = col; i < size_k; ++i, ++start_1, start_2 += size_n)
        acc += in_tensor_1[start_1] * in_tensor_2[start_2];
    out_tensor[(row * size_n) + col] = acc;
}}
"""
        self.shader_code2 = """
#version 450

layout (local_size_x = {local_size_x}, local_size_y = {local_size_y}, local_size_z = 1) in;
layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
layout (constant_id = 0) const float size_m_f = 0;
layout (constant_id = 1) const float size_k_f = 0;
layout (constant_id = 2) const float size_n_f = 0;
layout (constant_id = 3) const float size_b_f = 0;

void main()
{{
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint batch = gl_GlobalInvocationID.z;
    uint size_m = uint(size_m_f);
    uint size_k = uint(size_k_f);
    uint size_n = uint(size_n_f);
    uint size_b = uint(size_b_f);
    if(row >= size_m || col >= size_n || batch >= size_b) return;
    float acc = 0.0;
    uint start_1 = (batch * size_m * size_k) + (row * size_k);
    uint start_2 = (batch * size_k * size_n) + col;
    for(uint i = 0; i < size_k; ++i, ++start_1, start_2 += size_n)
        acc += in_tensor_1[start_1] * in_tensor_2[start_2];
    out_tensor[(batch * size_m * size_n) + (row * size_n) + col] = acc;
}}
"""

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulOp({device_name})"

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
            tensor_out = self.manager.tensor(np.zeros(rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols]:
                self.sizes = [rows, cols, ncols]
                local_size_x = min(self.local_size_x, rows)
                local_size_y = min(self.local_size_y, ncols)
                # compile shader
                self.shader = compile_source(
                    self.shader_code1.format(local_size_x=local_size_x, local_size_y=local_size_y))
                self.workgroup = (
                    (rows + local_size_x - 1) // local_size_x, (ncols + local_size_y - 1) // local_size_y, 1)

            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                                             self.shader, self.workgroup, self.sizes, []))

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
            tensor_out = self.manager.tensor(np.zeros(blocks * rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols, blocks]:
                self.sizes = [rows, cols, ncols, blocks]
                local_size_x = min(self.local_size_x, rows)
                local_size_y = min(self.local_size_y, ncols)
                # compile shader
                self.shader = compile_source(
                    self.shader_code2.format(local_size_x=local_size_x, local_size_y=local_size_y))
                self.workgroup = (
                    (rows + local_size_x - 1) // local_size_x, (ncols + local_size_y - 1) // local_size_y, blocks)

            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                                             self.shader, self.workgroup, self.sizes, []))

            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]
