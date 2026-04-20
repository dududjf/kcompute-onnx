import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class MaxUnpoolOp:

    def __init__(self, manager: kp.Manager, kernel_shape=None, pads=None, strides=None):
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.manager = manager
        self.scatter_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly buffer XBuf {{ float in_buf[]; }};
layout (std430, set = 0, binding = 1) readonly buffer IBuf {{ float idx_buf[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer YBuf {{ float out_buf[]; }};
layout (std430, set = 0, binding = 3) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    uint total_in = params[0];
    
    if(gid >= total_in) return;

    uint pos = uint(idx_buf[gid]);
    out_buf[pos] = in_buf[gid];
}}
""")
        self.copy_1d_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly buffer InBuf {{ float in_buf[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_buf[]; }};
layout (std430, set = 0, binding = 2) readonly buffer Params {{ uint params[]; }};

void main() {{
    uint pre_idx = gl_GlobalInvocationID.x;
    uint dim_idx = gl_GlobalInvocationID.y;
    uint post_idx = gl_GlobalInvocationID.z;

    uint pre_size = params[0];
    uint post_in = params[1];
    uint post_out = params[2];
    uint in_dim = params[3];
    uint out_dim = params[4];
    uint min_dim = params[5];

    if(pre_idx >= pre_size || dim_idx >= min_dim || post_idx >= post_in) return;

    uint src = pre_idx * in_dim * post_in + dim_idx * post_in + post_idx;
    uint dst = pre_idx * out_dim * post_out + dim_idx * post_out + post_idx;
    out_buf[dst] = in_buf[src];
}}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"MaxUnpoolOp({dev})"

    __str__ = __repr__

    @staticmethod
    def _inferred_spatial_shape(input_spatial_shape, kernel_shape, strides, pads):
        dims = len(input_spatial_shape)
        out = [0] * dims
        for d in range(dims):
            out[d] = (
                    (input_spatial_shape[d] - 1) * strides[d]
                    - (pads[d] + pads[dims + d])
                    + kernel_shape[d]
            )
        return out

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
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
        assert len(input_tensors) >= 2, "MaxUnpool expects two inputs: X and indices"
        tensor_in, shape_in = input_tensors[0]
        tensor_idx, shape_idx = input_tensors[1]
        assert len(shape_in) >= 3, f"Input must be not less than 3D, got {shape_in}"
        assert shape_in == shape_idx, "Indices shape must match input"

        n, c = shape_in[:2]
        spatial_dims = len(shape_in) - 2

        kernel = self.kernel_shape
        strides = self.strides or [1] * spatial_dims
        pads = self.pads or [0] * (2 * spatial_dims)

        inferred_shape = [n, c] + self._inferred_spatial_shape(shape_in[2:], kernel, strides, pads)

        # 解析 output_shape
        final_shape = inferred_shape
        if len(input_tensors) >= 3:
            out_shape_list = [int(v) for v in input_tensors[2][0].data().ravel()]
            if len(out_shape_list) == len(inferred_shape):
                final_shape = out_shape_list
            elif len(out_shape_list) == spatial_dims:
                final_shape = [n, c] + out_shape_list

        # Scatter到推断形状
        total_in = np.prod(shape_in)
        total_out_inferred = np.prod(inferred_shape)
        y_inferred = self.manager.tensor(np.zeros(total_out_inferred, dtype=np.float32))
        updated_tensors.append(y_inferred)

        params = np.array([total_in], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        group_x = (total_in + LOCAL_X_1D - 1) // LOCAL_X_1D

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_idx, y_inferred, param_in],
            self.scatter_shader,
            (group_x, 1, 1)
        ))

        # 若形状完全匹配，直接返回
        if final_shape == inferred_shape:
            return [(y_inferred, inferred_shape)]

        curr_tensor = y_inferred
        curr_shape = inferred_shape

        for d in range(spatial_dims):
            curr_dim = curr_shape[2 + d]
            target_dim = final_shape[2 + d]

            pre_size = int(n * c * np.prod(curr_shape[2:2 + d]))
            post_in = int(np.prod(curr_shape[3 + d:]))
            curr_shape[2 + d] = target_dim
            post_out = int(np.prod(curr_shape[3 + d:]))

            next_tensor = self.manager.tensor(np.zeros(int(np.prod(curr_shape)), dtype=np.float32))
            updated_tensors.append(next_tensor)

            min_dim = min(curr_dim, target_dim)
            params_copy = np.array([pre_size, post_in, post_out, curr_dim, target_dim, min_dim], dtype=np.uint32)
            param_in_copy = self.manager.tensor_t(params_copy, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in_copy])).eval()

            group_x = (pre_size + LOCAL_X_3D - 1) // LOCAL_X_3D
            group_y = (min_dim + LOCAL_Y_3D - 1) // LOCAL_Y_3D
            group_z = (post_in + LOCAL_Z_3D - 1) // LOCAL_Z_3D

            updated_algorithms.append(self.manager.algorithm(
                [curr_tensor, next_tensor, param_in_copy],
                self.copy_1d_shader,
                (group_x, group_y, group_z)
            ))

            curr_tensor = next_tensor

        tensor_out, shape_out = curr_tensor, curr_shape
        return [(tensor_out, shape_out)]

