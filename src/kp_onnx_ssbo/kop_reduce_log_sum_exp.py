import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D, LOCAL_X_1D


class ReduceLogSumExpOp:
    def __init__(self, manager: kp.Manager, keepdims=True, noop_with_empty_axes=False):
        self.manager = manager
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes

        # Max reduction along a single dimension
        self.compiled_shader_max = compile_source(f"""
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

    float max_val = in_tensor[in_offset];
    in_offset += block_size;
    for (uint i = 1; i < dimension; ++i, in_offset += block_size) {{
        max_val = max(max_val, in_tensor[in_offset]);
    }}
    out_tensor[out_offset] = max_val;
}}
""")

        # Sum reduction along a single dimension
        self.compiled_shader_sum = compile_source(f"""
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

        # sub_exp in 3D: out = exp(input - mx_broadcast)
        self.compiled_shader_sub_exp = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout (std430, set = 0, binding = 1) readonly  buffer MxBuf   {{ float mx[];         }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint max_x = params[0];
    uint max_y = params[1];
    uint max_z = params[2];
    if (gx >= max_x || gy >= max_y || gz >= max_z) return;

    uint sy_in = params[3];
    uint sz_in = params[4];
    uint sx_mx = params[5];
    uint sy_mx = params[6];
    uint sz_mx = params[7];

    uint p_in = (gx * sy_in + gy) * sz_in + gz;

    uint ax = min(gx, sx_mx - 1);
    uint ay = min(gy, sy_mx - 1);
    uint az = min(gz, sz_mx - 1);
    uint p_mx = (ax * sy_mx + ay) * sz_mx + az;

    out_tensor[p_in] = exp(in_tensor[p_in] - mx[p_mx]);
}}
""")

        # y = log(x) + m
        self.compiled_shader_log_add = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer SumBuf  {{ float sum_[];  }};
layout (std430, set = 0, binding = 1) readonly  buffer MxBuf   {{ float mx[];    }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf  {{ float out_[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint total_size = params[0];
    if (gx >= total_size) return;

    out_[gx] = log(sum_[gx]) + mx[gx];
}}
""")

    def __repr__(self):
        return f"ReduceLogSumExpOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        if inputs[0].size == 0:
            input_tensors.append((None, []))
        else:
            input_tensors.append((self.manager.tensor(inputs[0]), list(inputs[0].shape)))
        if len(inputs) > 1:
            if inputs[1] is not None:
                numpy_in = inputs[1].reshape(-1).astype(np.float32) \
                    if isinstance(inputs[1], np.ndarray) else np.array(inputs[1], dtype=np.float32)
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append(
                    (tensor, list(inputs[1].shape) if isinstance(inputs[1], np.ndarray) else [len(inputs[1])]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            real_inputs = [] if input_tensors[0][0] is None else [input_tensors[0][0]]
            seq.record(kp.OpTensorSyncDevice(real_inputs + updated_tensors))
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

        if self.noop_with_empty_axes and axes is None:
            return [input_tensors[0]]

        tensor_in = input_tensors[0][0]
        ori_tensor_in = tensor_in
        shape_in = input_tensors[0][1]

        # x.size == 0 → constant -inf with reduced shape
        if tensor_in is None:
            shape_out = [1]
            numpy_out = np.full(int(np.prod(shape_out)), -np.inf, dtype=np.float32)
            tensor_out = self.manager.tensor(numpy_out)
            updated_tensors.append(tensor_out)
            return [(tensor_out, shape_out)]

        # mark reduced axes
        if axes is None:
            axis_present = [True] * len(shape_in)
        else:
            axis_present = [False] * len(shape_in)
            for axis in axes:
                idx = axis if axis >= 0 else axis + len(shape_in)
                axis_present[idx] = True

        # compute shape_out (no-keepdims)
        shape_out_nk = [shape_in[i] for i in range(len(shape_in)) if not axis_present[i]]
        if self.keepdims:
            shape_out = [1 if axis_present[i] else shape_in[i] for i in range(len(shape_in))]
        else:
            shape_out = shape_out_nk

        # 1) reduce max across all selected axes → mx (shape_out_nk)
        tensor_max = tensor_in
        block_size = 1
        for i in reversed(range(len(shape_in))):
            if axis_present[i] and shape_in[i] > 1:
                group_x = int(np.prod(shape_in[:i])) if i > 0 else 1
                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                prev_in = tensor_max
                tensor_max = self.manager.tensor(numpy_out)
                updated_tensors.append(tensor_max)

                params = np.array([group_x, block_size, shape_in[i]], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                workgroup = (
                    (group_x + LOCAL_X_2D - 1) // LOCAL_X_2D,
                    (block_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                    1,
                )

                updated_algorithms.append(self.manager.algorithm(
                    [prev_in, tensor_max, param_in],
                    self.compiled_shader_max,
                    workgroup,
                ))
            else:
                block_size *= int(shape_in[i])

        # Compute 3D sizes for sub_exp shader
        if len(shape_in) == 1:
            sx_in, sy_in, sz_in = shape_in[0], 1, 1
        elif len(shape_in) == 2:
            sx_in, sy_in, sz_in = shape_in[0], shape_in[1], 1
        else:
            sx_in, sy_in, sz_in = int(np.prod(shape_in[:-2])), shape_in[-2], shape_in[-1]

        if len(shape_out_nk) == 0:
            sx_mx, sy_mx, sz_mx = 1, 1, 1
        elif len(shape_out_nk) == 1:
            sx_mx, sy_mx, sz_mx = shape_out_nk[0], 1, 1
        elif len(shape_out_nk) == 2:
            sx_mx, sy_mx, sz_mx = shape_out_nk[0], shape_out_nk[1], 1
        else:
            sx_mx, sy_mx, sz_mx = int(np.prod(shape_out_nk[:-2])), shape_out_nk[-2], shape_out_nk[-1]

        # 2) sub-and-exp to full-size buffer
        size_in = int(np.prod(shape_in))
        tensor_exp = self.manager.tensor(np.zeros(size_in, dtype=np.float32))
        updated_tensors.append(tensor_exp)

        params_sub_exp = np.array([sx_in, sy_in, sz_in, sy_in, sz_in, sx_mx, sy_mx, sz_mx], dtype=np.uint32)
        param_sub_exp_in = self.manager.tensor_t(params_sub_exp, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_sub_exp_in])).eval()

        workgroup_sub_exp = (
            (sx_in + LOCAL_X_3D - 1) // LOCAL_X_3D,
            (sy_in + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
            (sz_in + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
        )

        updated_algorithms.append(self.manager.algorithm(
            [ori_tensor_in, tensor_max, tensor_exp, param_sub_exp_in],
            self.compiled_shader_sub_exp,
            workgroup_sub_exp,
        ))

        # 3) reduce sum across axes on tensor_exp → sum_exp shape_out_nk
        tensor_sum = tensor_exp
        block_size = 1
        for i in reversed(range(len(shape_in))):
            if axis_present[i] and shape_in[i] > 1:
                group_x = int(np.prod(shape_in[:i])) if i > 0 else 1
                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                prev_in = tensor_sum
                tensor_sum = self.manager.tensor(numpy_out)
                updated_tensors.append(tensor_sum)

                params = np.array([group_x, block_size, shape_in[i]], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                workgroup = (
                    (group_x + LOCAL_X_2D - 1) // LOCAL_X_2D,
                    (block_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                    1,
                )

                updated_algorithms.append(self.manager.algorithm(
                    [prev_in, tensor_sum, param_in],
                    self.compiled_shader_sum,
                    workgroup,
                ))
            else:
                block_size *= int(shape_in[i])

        # 4) out = log(sum_exp) + mx
        out_len = int(np.prod(shape_out_nk)) if len(shape_out_nk) > 0 else 1
        tensor_out = self.manager.tensor(np.zeros(out_len, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params_log_add = np.array([out_len], dtype=np.uint32)
        param_log_add_in = self.manager.tensor_t(params_log_add, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_log_add_in])).eval()

        workgroup_log_add = ((out_len + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_sum, tensor_max, tensor_out, param_log_add_in],
            self.compiled_shader_log_add,
            workgroup_log_add,
        ))

        return [(tensor_out, shape_out)]
