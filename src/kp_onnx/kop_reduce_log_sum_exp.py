import numpy as np
import kp
from .shader_utils import compile_source


class ReduceLogSumExpOp:
    def __init__(self, manager: kp.Manager, keepdims=True, noop_with_empty_axes=False):
        self.manager = manager
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes

        # Max reduction along a single dimension
        self.compiled_shader_max = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) readonly buffer buf_in  { float in_tensor[];  };
layout (binding = 1) buffer        buf_out { float out_tensor[]; };

layout (constant_id = 0) const float dimension_f  = 0;
layout (constant_id = 1) const float block_size_f = 0;
void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);
    uint in_offset  = gx * dimension * block_size + gy;
    uint out_offset = gx * block_size + gy;
    
    float max_val = in_tensor[in_offset++];
    for (uint i = 1; i < dimension; ++i, in_offset += block_size) {
        max_val = max(max_val, in_tensor[in_offset]);
    }
    out_tensor[out_offset] = max_val;
}
""")

        # Sum reduction along a single dimension
        self.compiled_shader_sum = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) readonly buffer buf_in  { float in_tensor[];  };
layout (binding = 1) buffer        buf_out { float out_tensor[]; };

layout (constant_id = 0) const float dimension_f  = 0;
layout (constant_id = 1) const float block_size_f = 0;
void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);
    uint in_offset  = gx * dimension * block_size + gy;
    uint out_offset = gx * block_size + gy;
    float acc = 0.0;
    for (uint i = 0; i < dimension; ++i, in_offset += block_size) {
        acc += in_tensor[in_offset];
    }
    out_tensor[out_offset] = acc;
}
""")

        # sub_exp in 3D without div/mod: out = exp(input - mx_broadcast)
        self.compiled_shader_sub_exp = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) readonly buffer buf_in  { float in_tensor[];  };
layout (binding = 1) readonly buffer buf_mx  { float mx[];         };
layout (binding = 2) buffer        buf_out { float out_tensor[];  };
layout (constant_id = 0) const float sy_in_f = 0;
layout (constant_id = 1) const float sz_in_f = 0;
layout (constant_id = 2) const float sx_mx_f = 0;
layout (constant_id = 3) const float sy_mx_f = 0;
layout (constant_id = 4) const float sz_mx_f = 0;
void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    uint sy_in = uint(sy_in_f);
    uint sz_in = uint(sz_in_f);
    uint sx_mx = uint(sx_mx_f);
    uint sy_mx = uint(sy_mx_f);
    uint sz_mx = uint(sz_mx_f);
    uint p_in = (gx * sy_in + gy) * sz_in + gz;
    
    uint ax = min(gx, sx_mx - 1);
    uint ay = min(gy, sy_mx - 1);
    uint az = min(gz, sz_mx - 1);
    uint p_mx = (ax * sy_mx + ay) * sz_mx + az;
    out_tensor[p_in] = exp(in_tensor[p_in] - mx[p_mx]);
}
""")

        # y = log(x) + m
        self.compiled_shader_log_add = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) readonly buffer buf_sum { float sum_[]; };
layout (binding = 1) readonly buffer buf_mx  { float mx[];    };
layout (binding = 2) buffer        buf_out { float out_[];  };
void main()
{
    uint gx = gl_GlobalInvocationID.x;
    out_[gx] = log(sum_[gx]) + mx[gx];
}
""")

    def __repr__(self):
        return f"ReduceLogSumExpOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

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
            # sync only the real input if present, plus our created buffers
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
                workgroup = (group_x, block_size, 1)
                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                prev_in = tensor_max
                tensor_max = self.manager.tensor(numpy_out)
                updated_algorithms.append(self.manager.algorithm(
                    [prev_in, tensor_max],
                    self.compiled_shader_max,
                    workgroup,
                    [shape_in[i], block_size],
                    []
                ))
                updated_tensors.append(tensor_max)
            else:
                block_size *= int(shape_in[i])

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
        updated_algorithms.append(self.manager.algorithm(
            [ori_tensor_in, tensor_max, tensor_exp],
            self.compiled_shader_sub_exp,
            (sx_in, sy_in, sz_in),
            [sy_in, sz_in, sx_mx, sy_mx, sz_mx],
            []
        ))
        updated_tensors.append(tensor_exp)

        # 3) reduce sum across axes on tensor_exp → sum_exp shape_out_nk
        tensor_sum = tensor_exp
        block_size = 1
        for i in reversed(range(len(shape_in))):
            if axis_present[i] and shape_in[i] > 1:
                group_x = int(np.prod(shape_in[:i])) if i > 0 else 1
                workgroup = (group_x, block_size, 1)
                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                prev_in = tensor_sum
                tensor_sum = self.manager.tensor(numpy_out)
                updated_algorithms.append(self.manager.algorithm(
                    [prev_in, tensor_sum],
                    self.compiled_shader_sum,
                    workgroup,
                    [shape_in[i], block_size],
                    []
                ))
                updated_tensors.append(tensor_sum)
            else:
                block_size *= int(shape_in[i])

        # 4) out = log(sum_exp) + mx  (both shape_out_nk, equal length)
        out_len = int(np.prod(shape_out_nk)) if len(shape_out_nk) > 0 else 1
        tensor_out = self.manager.tensor(np.zeros(out_len, dtype=np.float32))
        updated_algorithms.append(self.manager.algorithm([tensor_sum, tensor_max, tensor_out],
                                                         self.compiled_shader_log_add))
        updated_tensors.append(tensor_out)
        return [(tensor_out, shape_out)]
