import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class AffineGridOp:
    def __init__(self, manager: kp.Manager, align_corners=0):
        self.manager = manager
        self.align_corners = align_corners
        self.compiled_shader_transpose = compile_source(f"""
#version 450
layout(local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_buf[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};
layout (constant_id = 0) const float pre_block_size_f = 0;
layout (constant_id = 1) const float post_block_size_f = 0;
layout (constant_id = 2) const float axis_dimension_f = 0;

void main() 
{{
    uint leading_size = params[0], pre_block_size = params[1], axis_dimension = params[2], post_block_size = params[3];
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    if(gx >= leading_size || gy >= pre_block_size || gz >= axis_dimension) return;

    uint stride_y = axis_dimension * post_block_size;
    uint stride_x = pre_block_size * stride_y;
    uint in_index = gx * stride_x + gy * stride_y + gz * post_block_size;
    uint out_index = gx * stride_x + gz * pre_block_size * post_block_size + gy * post_block_size;

    for(uint i = 0; i < post_block_size; ++i, ++out_index, ++in_index)
        out_buf[out_index] = in_buf[in_index];
}}""")

        self.compiled_shader_matmul = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf1 {{ float in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer InBuf2 {{ float in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main()
{{
    uint size_m = params[0], size_n = params[1], size_k = params[2];
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    if (row >= size_m ||  col >= size_n) return;
    float acc = 0.0;
    uint start_1 = row * size_k;
    for(uint i = 0, start_2 = col; i < size_k; i++, start_1++, start_2 += size_n)
        acc += in_tensor_1[start_1] * in_tensor_2[start_2];
    out_tensor[row * size_n + col] = acc;
}}
""")

        self.compile_shader_stack = compile_source(f"""
#version 450
layout(local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_tensor[];  }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_tensor[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main()
{{
    uint group_count = params[0], axis_dim = params[1], block_size = params[2];
    uint out_axis_offset = params[3], out_axis_dim = params[4];
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    if(gx >= group_count || gy >= axis_dim) return;

    uint in_offset = gx * axis_dim * block_size + gy * block_size;
    uint out_offset = gx * out_axis_dim * block_size + (out_axis_offset + gy) * block_size;

    for (uint i = 0; i < block_size; ++i, ++in_offset, ++out_offset) {{
        out_tensor[out_offset] = in_tensor[in_offset];
    }}
}}
""")

    def __repr__(self):
        return f"AffineGridOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(shape_out)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        theta_tensor, theta_shape = input_tensors[0]
        size = input_tensors[1][0].data().astype(int)
        align_corners = self.align_corners

        assert len(theta_shape) == 3, "theta shall have shape of (N, 2, 3) for 2D, (N, 3, 4) for 3D"

        # Generate base grid coordinates  Generate normalized coordinates
        if len(size) == 4:
            H, W = size[2], size[3]
            original_grid = [np.ones([H, W])]
            size_zeros = np.zeros([H, W], dtype=np.float32)
            if align_corners == 1:
                a1 = np.arange(-1, 1.0001, 2.0 / (H - 1))
                y = np.reshape(a1, (H, 1)) + size_zeros
                original_grid = [y, *original_grid]
                a2 = np.arange(-1, 1.0001, 2.0 / (W - 1))
                x = np.reshape(a2, (1, W)) + size_zeros
                original_grid = [x, *original_grid]
            else:
                a1 = np.arange(-1.0 + 1.0 / H, 1, 2.0 / H)
                y = np.reshape(a1, (H, 1)) + size_zeros
                original_grid = [y, *original_grid]
                a2 = np.arange(-1.0 + 1.0 / W, 1, 2.0 / W)
                x = np.reshape(a2, (1, W)) + size_zeros
                original_grid = [x, *original_grid]
        else:
            D, H, W = size[2], size[3], size[4]
            original_grid = [np.ones([D, H, W])]
            size_zeros = np.zeros([D, H, W], dtype=np.float32)
            if align_corners == 1:
                a1 = np.arange(-1, 1.0001, 2.0 / (D - 1))
                z = np.reshape(a1, (D, 1, 1)) + size_zeros
                original_grid = [z, *original_grid]
                a2 = np.arange(-1, 1.0001, 2.0 / (H - 1))
                y = np.reshape(a2, (1, H, 1)) + size_zeros
                original_grid = [y, *original_grid]
                a3 = np.arange(-1, 1.0001, 2.0 / (W - 1))
                x = np.reshape(a3, (1, W)) + size_zeros
                original_grid = [x, *original_grid]
            else:
                a1 = np.arange(-1.0 + 1.0 / D, 1, 2.0 / D)
                z = np.reshape(a1, (D, 1, 1)) + size_zeros
                original_grid = [z, *original_grid]
                a2 = np.arange(-1.0 + 1.0 / H, 1, 2.0 / H)
                y = np.reshape(a2, (1, H, 1)) + size_zeros
                original_grid = [y, *original_grid]
                a3 = np.arange(-1.0 + 1.0 / W, 1, 2.0 / W)
                x = np.reshape(a3, (1, W)) + size_zeros
                original_grid = [x, *original_grid]

        stack_tensors = []
        for ori in original_grid:
            tensor = self.manager.tensor(ori.reshape(-1).astype(np.float32))
            updated_tensors.append(tensor)
            shape = [s for s in size[2:]] + [1]
            stack_tensors.append((tensor, shape))

        shape_out = stack_tensors[0][1].copy()
        shape_out[-1] *= len(stack_tensors)
        tensor_concat = self.manager.tensor(np.zeros(np.prod(shape_out), dtype=np.float32))
        updated_tensors.append(tensor_concat)
        group_count = int(np.prod(shape_out[:-1]))
        block_size = 1
        offset = 0
        for tensor, shape in stack_tensors:
            axis_dim = shape[-1]
            params = [group_count, axis_dim, block_size, offset, shape_out[-1]]
            param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
            group_x = (group_count + LOCAL_X_2D - 1) // LOCAL_X_2D
            group_y = (axis_dim + LOCAL_Y_2D - 1) // LOCAL_Y_2D
            workgroup = (group_x, group_y, 1)
            updated_algorithms.append(self.manager.algorithm([tensor, tensor_concat, param_in],
                                                             self.compile_shader_stack,
                                                             workgroup))
            offset += axis_dim

        if len(shape_out) == 3:  # 2D case
            H, W, dim_homo = shape_out
            transposed_tensor1 = self._apply_transpose(tensor_concat, [H * W, dim_homo], [1, 0],
                                                       updated_algorithms, updated_tensors)
            grid_n = self._apply_matmul(theta_tensor, theta_shape, transposed_tensor1, [dim_homo, H * W],
                                        updated_algorithms, updated_tensors)
            transposed_tensor2 = self._apply_transpose(grid_n, [theta_shape[0], theta_shape[1], H * W], [0, 2, 1],
                                                       updated_algorithms, updated_tensors)
            shape_out = [theta_shape[0], H, W, theta_shape[1]]

        else:  # 3D case
            D, H, W, dim_homo = shape_out
            transposed_tensor1 = self._apply_transpose(tensor_concat, [D * H * W, dim_homo], [1, 0],
                                                       updated_algorithms, updated_tensors)
            grid_n = self._apply_matmul(theta_tensor, theta_shape, transposed_tensor1, [dim_homo, D * H * W],
                                        updated_algorithms, updated_tensors)
            transposed_tensor2 = self._apply_transpose(grid_n, [theta_shape[0], theta_shape[1], D * H * W], [0, 2, 1],
                                                       updated_algorithms, updated_tensors)
            shape_out = [theta_shape[0], D, H, W, theta_shape[1]]

        return [(transposed_tensor2, shape_out)]

    def _apply_transpose(self, input_tensor, shape, perm_vals, updated_algorithms, updated_tensors):
        tensor_out = input_tensor
        ndim = len(shape)
        shape_out = [shape[i] for i in perm_vals]
        total_size = np.prod(shape_out)
        leading_size = 1
        suffix = list(range(ndim))
        for i in range(ndim - 1):
            if suffix[0] != perm_vals[i]:
                tensor_in = tensor_out
                tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
                pre_block_size = shape[suffix[0]]
                post_block_size = 1
                axis_dimension = shape_out[i]

                params = [leading_size, pre_block_size, axis_dimension, post_block_size]
                param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
                group_x = (leading_size + LOCAL_X_3D - 1) // LOCAL_X_3D
                group_y = (pre_block_size + LOCAL_Y_3D - 1) // LOCAL_Y_3D
                group_z = (axis_dimension + LOCAL_Z_3D - 1) // LOCAL_Z_3D
                workgroup = (group_x, group_y, group_z)
                updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out, param_in],
                                                                 self.compiled_shader_transpose,
                                                                 workgroup))
                updated_tensors.append(tensor_out)
            suffix.remove(perm_vals[i])
            leading_size *= shape_out[i]
        return tensor_out

    def _apply_matmul(self, theta_tensor, theta_shape, grid_tensor, grid_shape, updated_algorithms, updated_tensors):
        rows = np.prod(theta_shape[:-1])
        cols = theta_shape[-1]
        ncols = grid_shape[1]

        tensor_out = self.manager.tensor(np.zeros(rows * ncols, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params = [rows, ncols, cols]
        param_in = self.manager.tensor_t(np.array(params, dtype=np.uint32), kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
        group_x = (rows + LOCAL_X_2D - 1) // LOCAL_X_2D
        group_y = (ncols + LOCAL_Y_2D - 1) // LOCAL_Y_2D
        workgroup = (group_x, group_y, 1)

        updated_algorithms.append(self.manager.algorithm([theta_tensor, grid_tensor, tensor_out, param_in],
                                                         self.compiled_shader_matmul,
                                                         workgroup))
        return tensor_out
