import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to, LOCAL_X_1D, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class EinsumOp:
    """
    将任意 einsum 表达式分解为基本操作序列：
    - 对角线提取
    - ReduceSum
    - Transpose
    - Unsqueeze
    - Mul
    - MatMul
    """

    def __init__(self, manager: kp.Manager, equation: str = ""):
        self.manager = manager
        self.equation = equation

        # 用于广播的逐元素乘法的 Shader
        self.shader_mul = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf1  {{ float in1[];      }};
layout(std430, set = 0, binding = 1) readonly  buffer InBuf2  {{ float in2[];      }};
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf  {{ float out_data[]; }};
layout(std430, set = 0, binding = 3) readonly  buffer Params  {{
    uint size_x_1;
    uint size_y_1;
    uint size_z_1;
    uint size_x_2;
    uint size_y_2;
    uint size_z_2;
    uint stride_x_1;
    uint stride_y_1;
    uint stride_x_2;
    uint stride_y_2;
    uint stride_x;
    uint stride_y;
    uint out_dim_x;
    uint out_dim_y;
    uint out_dim_z;
}};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    if (gx >= out_dim_x || gy >= out_dim_y || gz >= out_dim_z) return;

    uint x_1 = min(gx, size_x_1 - 1u);
    uint y_1 = min(gy, size_y_1 - 1u);
    uint z_1 = min(gz, size_z_1 - 1u);
    uint x_2 = min(gx, size_x_2 - 1u);
    uint y_2 = min(gy, size_y_2 - 1u);
    uint z_2 = min(gz, size_z_2 - 1u);

    out_data[gx * stride_x + gy * stride_y + gz] =
        in1[x_1 * stride_x_1 + y_1 * stride_y_1 + z_1] *
        in2[x_2 * stride_x_2 + y_2 * stride_y_2 + z_2];
}}
""")

        # 沿轴进行 ReduceSum 的 Shader
        self.shader_reduce = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer Params {{
    uint outer_size;
    uint reduce_size;
    uint inner_size;
}};

void main() {{
    uint outer = gl_GlobalInvocationID.x;
    uint inner = gl_GlobalInvocationID.y;

    if (outer >= outer_size || inner >= inner_size) return;

    uint base = outer * reduce_size * inner_size + inner;

    float sum = 0.0;
    for (uint i = 0; i < reduce_size; ++i) {{
        sum += in_data[base];
        base += inner_size;
    }}

    out_data[outer * inner_size + inner] = sum;
}}
""")

        # 全局 ReduceSum 的专用 Shader（将整个张量求和为标量）
        self.shader_reduce_all = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer Params {{
    uint total_size;
}};

void main() {{
    if (gl_GlobalInvocationID.x > 0u) return;

    float sum = 0.0;
    for (uint i = 0u; i < total_size; ++i) {{
        sum += in_data[i];
    }}
    out_data[0] = sum;
}}
""")

        # 单轴交换的转置 Shader
        self.shader_transpose = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer Params {{
    uint pre_block_size;
    uint post_block_size;
    uint axis_dimension;
    uint stride_y;
    uint stride_x;
    uint pre_post;
    uint leading_size;
}};

void main() {{
    uint gy = gl_GlobalInvocationID.x;   // pre_block_size
    uint gx = gl_GlobalInvocationID.y;   // leading_size
    uint gz = gl_GlobalInvocationID.z;   // axis_dimension

    if (gy >= pre_block_size || gx >= leading_size || gz >= axis_dimension) return;

    uint in_index  = gx * stride_x + gy * stride_y + gz * post_block_size;
    uint out_index = gx * stride_x + gz * pre_post  + gy * post_block_size;

    for (uint i = 0u; i < post_block_size; ++i) {{
        out_data[out_index + i] = in_data[in_index + i];
    }}
}}
""")

        # 对角线提取的 Shader
        self.shader_diagonal = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf  {{ float in_data[];  }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer Params {{
    uint outer_size;
    uint diag_size;
    uint inner_size;
    uint stride;
    uint in_outer_stride;
    uint diag_stride;
    uint out_row_stride;
}};

void main() {{
    uint outer    = gl_GlobalInvocationID.x;
    uint diag_idx = gl_GlobalInvocationID.y;
    uint inner    = gl_GlobalInvocationID.z;

    if (outer >= outer_size || diag_idx >= diag_size || inner >= inner_size) return;

    uint in_base  = outer * in_outer_stride + diag_idx * diag_stride + inner;
    uint out_base = outer * out_row_stride  + diag_idx * inner_size  + inner;

    out_data[out_base] = in_data[in_base];
}}
""")

        # 分批矩阵乘法的 Shader
        self.shader_matmul = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf1 {{ float in1[];      }};
layout(std430, set = 0, binding = 1) readonly  buffer InBuf2 {{ float in2[];      }};
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf {{ float out_data[]; }};
layout(std430, set = 0, binding = 3) readonly  buffer Params {{
    uint batch_size;
    uint M;
    uint K;
    uint N;
    uint a_batch_stride;
    uint b_batch_stride;
    uint out_batch_stride;
}};

void main() {{
    uint row   = gl_GlobalInvocationID.x;
    uint col   = gl_GlobalInvocationID.y;
    uint batch = gl_GlobalInvocationID.z;

    if (batch >= batch_size || row >= M || col >= N) return;

    uint a_base = batch * a_batch_stride + row * K;
    uint b_base = batch * b_batch_stride + col;

    float sum = 0.0;
    for (uint k = 0u; k < K; ++k) {{
        sum += in1[a_base + k] * in2[b_base + k * N];
    }}

    out_data[batch * out_batch_stride + row * N + col] = sum;
}}
""")

        # 通用对角线提取 Shader
        self.shader_diagonal_general = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBuf       {{ float in_data[];   }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf      {{ float out_data[];  }};
layout(std430, set = 0, binding = 2) readonly  buffer IndexMapBuf {{ float index_map[]; }};
layout(std430, set = 0, binding = 3) readonly  buffer Params      {{
    uint out_size;
}};

void main() {{
    uint out_idx = gl_GlobalInvocationID.x;
    if (out_idx >= out_size) return;

    uint in_idx = uint(index_map[out_idx]);
    out_data[out_idx] = in_data[in_idx];
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"EinsumOp({device_name}, equation='{self.equation}')"

    __str__ = __repr__

    def run(self, *inputs):
        if any(0 in inp.shape for inp in inputs):
            return [np.einsum(self.equation, *inputs).astype(np.float32)]

        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        all_tensors_to_sync = [t[0] for t in input_tensors] + updated_tensors
        seq.record(kp.OpTensorSyncDevice(all_tensors_to_sync))
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

        def make_param(arr):
            """创建 uint32 参数 tensor 并立即同步到 GPU。"""
            t = self.manager.tensor_t(np.array(arr, dtype=np.uint32), kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([t])).eval()
            return t

        def reduce_unused(tensor, shape, subscripts, keep_subs):
            """化简（求和）不在 keep_subs 中的轴"""
            reduce_indices = [i for i, s in enumerate(subscripts) if s not in keep_subs]

            if not reduce_indices:
                return tensor, shape, subscripts

            if len(reduce_indices) == len(shape):
                total_size = int(np.prod(shape))
                out_tensor = self.manager.tensor(np.zeros(1, dtype=np.float32))
                updated_tensors.append(out_tensor)

                param_in = make_param([total_size])
                updated_algorithms.append(self.manager.algorithm(
                    [tensor, out_tensor, param_in],
                    self.shader_reduce_all,
                    (1, 1, 1)
                ))
                return out_tensor, [], []

            # 逐个化简轴（从最高索引到最低）
            for idx in sorted(reduce_indices, reverse=True):
                if shape[idx] == 1:
                    shape = shape[:idx] + shape[idx + 1:]
                    subscripts = subscripts[:idx] + subscripts[idx + 1:]
                else:
                    outer_size = int(np.prod(shape[:idx])) if idx > 0 else 1
                    reduce_size = shape[idx]
                    inner_size = int(np.prod(shape[idx + 1:])) if idx < len(shape) - 1 else 1

                    out_size = outer_size * inner_size
                    out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
                    updated_tensors.append(out_tensor)

                    param_in = make_param([outer_size, reduce_size, inner_size])
                    workgroup = (
                        (outer_size + LOCAL_X_2D - 1) // LOCAL_X_2D,
                        (inner_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                        1
                    )
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor, out_tensor, param_in],
                        self.shader_reduce,
                        workgroup
                    ))

                    tensor = out_tensor
                    shape = shape[:idx] + shape[idx + 1:]
                    subscripts = subscripts[:idx] + subscripts[idx + 1:]

            return tensor, shape, subscripts

        def transpose_axes(tensor, shape, perm):
            """根据排列转置张量"""
            if perm == list(range(len(perm))):
                return tensor, shape

            out_shape = [shape[i] for i in perm]
            total_size = int(np.prod(out_shape))
            leading_size = 1
            suffix = list(range(len(shape)))

            current_tensor = tensor
            for i in range(len(perm) - 1):
                if suffix[0] != perm[i]:
                    out_tensor = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
                    updated_tensors.append(out_tensor)

                    pre_block_size = shape[suffix[0]]
                    j = 1
                    while suffix[j] != perm[i]:
                        pre_block_size *= shape[suffix[j]]
                        j += 1

                    post_block_size = 1
                    j += 1
                    while j < len(suffix):
                        post_block_size *= shape[suffix[j]]
                        j += 1

                    axis_dimension = out_shape[i]
                    stride_y = axis_dimension * post_block_size
                    stride_x = pre_block_size * stride_y
                    pre_post = pre_block_size * post_block_size

                    param_in = make_param([
                        pre_block_size, post_block_size, axis_dimension,
                        stride_y, stride_x, pre_post, leading_size
                    ])
                    workgroup = (
                        (pre_block_size + LOCAL_X_3D - 1) // LOCAL_X_3D,
                        (leading_size   + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                        (axis_dimension + LOCAL_Z_3D - 1) // LOCAL_Z_3D
                    )
                    updated_algorithms.append(self.manager.algorithm(
                        [current_tensor, out_tensor, param_in],
                        self.shader_transpose,
                        workgroup
                    ))
                    current_tensor = out_tensor

                suffix.remove(perm[i])
                leading_size *= out_shape[i]

            return current_tensor, out_shape

        def multiply_broadcast(input1, input2):
            """带广播的逐元素乘法"""
            shape1 = input1['shape']
            shape2 = input2['shape']
            subs1  = input1['subscripts']
            subs2  = input2['subscripts']

            combined_subs = []
            for s in subs1:
                if s not in combined_subs:
                    combined_subs.append(s)
            for s in subs2:
                if s not in combined_subs:
                    combined_subs.append(s)

            out_shape = []
            for s in combined_subs:
                if s in subs1 and s in subs2:
                    d1 = shape1[subs1.index(s)]
                    d2 = shape2[subs2.index(s)]
                    assert d1 == d2 or d1 == 1 or d2 == 1, \
                        f"Incompatible dimensions for subscript '{s}': {d1} vs {d2}"
                    out_shape.append(max(d1, d2))
                elif s in subs1:
                    out_shape.append(shape1[subs1.index(s)])
                else:
                    out_shape.append(shape2[subs2.index(s)])

            bcast_shape1 = [shape1[subs1.index(s)] if s in subs1 else 1 for s in combined_subs]
            bcast_shape2 = [shape2[subs2.index(s)] if s in subs2 else 1 for s in combined_subs]

            tensor1 = input1['tensor']
            tensor2 = input2['tensor']

            if len(bcast_shape1) > 3:
                batch_out = out_shape[:-2]
                if bcast_shape2[:-2] != batch_out and not all(e == 1 for e in bcast_shape2[:-2]):
                    final_shape2 = batch_out + bcast_shape2[-2:]
                    tensor2 = broadcast_to(tensor2, bcast_shape2, final_shape2,
                                           updated_algorithms, updated_tensors, self.manager)
                    bcast_shape2 = final_shape2

            ndim = len(bcast_shape1)
            if ndim == 0:
                bcast_shape1 = [1, 1, 1]; bcast_shape2 = [1, 1, 1]; out_shape_3d = (1, 1, 1)
            elif ndim == 1:
                bcast_shape1 = bcast_shape1 + [1, 1]; bcast_shape2 = bcast_shape2 + [1, 1]
                out_shape_3d = (out_shape[0], 1, 1)
            elif ndim == 2:
                bcast_shape1 = bcast_shape1 + [1]; bcast_shape2 = bcast_shape2 + [1]
                out_shape_3d = (out_shape[0], out_shape[1], 1)
            elif ndim == 3:
                out_shape_3d = (out_shape[0], out_shape[1], out_shape[2])
            else:
                b1 = int(np.prod(bcast_shape1[:-2])); bcast_shape1 = [b1] + bcast_shape1[-2:]
                b2 = int(np.prod(bcast_shape2[:-2])); bcast_shape2 = [b2] + bcast_shape2[-2:]
                bo = int(np.prod(out_shape[:-2]))
                out_shape_3d = (bo, out_shape[-2], out_shape[-1])

            out_size = int(np.prod(out_shape)) if out_shape else 1
            out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
            updated_tensors.append(out_tensor)

            stride_x_1 = bcast_shape1[1] * bcast_shape1[2]
            stride_y_1 = bcast_shape1[2]
            stride_x_2 = bcast_shape2[1] * bcast_shape2[2]
            stride_y_2 = bcast_shape2[2]
            stride_x   = max(bcast_shape1[1], bcast_shape2[1]) * max(bcast_shape1[2], bcast_shape2[2])
            stride_y   = max(bcast_shape1[2], bcast_shape2[2])

            param_in = make_param(
                bcast_shape1 + bcast_shape2 +
                [stride_x_1, stride_y_1, stride_x_2, stride_y_2, stride_x, stride_y] +
                list(out_shape_3d)
            )
            workgroup = (
                (out_shape_3d[0] + LOCAL_X_3D - 1) // LOCAL_X_3D,
                (out_shape_3d[1] + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                (out_shape_3d[2] + LOCAL_Z_3D - 1) // LOCAL_Z_3D
            )
            updated_algorithms.append(self.manager.algorithm(
                [tensor1, tensor2, out_tensor, param_in],
                self.shader_mul,
                workgroup
            ))

            return {'tensor': out_tensor, 'shape': out_shape, 'subscripts': combined_subs}

        def contract_two(input1, input2, output_subs):
            """使用 mul 或 matmul 缩约两个输入"""
            subs1 = input1['subscripts']
            subs2 = input2['subscripts']
            subs1_set = set(subs1)
            subs2_set = set(subs2)

            shared = [s for s in subs1 if s in subs2_set]

            if not shared:
                return multiply_broadcast(input1, input2)

            output_set   = set(output_subs)
            contract_dims = [s for s in shared if s not in output_set]
            unshared1     = [s for s in subs1 if s not in subs2_set]
            unshared2     = [s for s in subs2 if s not in subs1_set]

            is_matmul = (len(contract_dims) >= 1 and
                         len(unshared1) >= 1 and len(unshared2) >= 1 and
                         all(s in output_set for s in unshared1) and
                         all(s in output_set for s in unshared2))

            if is_matmul:
                shape1 = input1['shape']
                shape2 = input2['shape']
                batch_subs    = [s for s in subs1 if s in subs2 and s in output_set]
                contract_subs = [s for s in shared if s not in output_set]

                batch_size = 1
                for s in batch_subs:
                    batch_size *= shape1[subs1.index(s)]

                M = 1
                for s in unshared1:
                    M *= shape1[subs1.index(s)]

                K = 1
                for s in contract_subs:
                    K *= shape1[subs1.index(s)]

                N = 1
                for s in unshared2:
                    N *= shape2[subs2.index(s)]

                desired_subs1 = batch_subs + unshared1 + contract_subs
                desired_subs2 = batch_subs + contract_subs + unshared2

                tensor1 = input1['tensor']
                if subs1 != desired_subs1:
                    perm1 = [subs1.index(s) for s in desired_subs1]
                    tensor1, _ = transpose_axes(tensor1, shape1, perm1)

                tensor2 = input2['tensor']
                if subs2 != desired_subs2:
                    perm2 = [subs2.index(s) for s in desired_subs2]
                    tensor2, _ = transpose_axes(tensor2, shape2, perm2)

                out_size = batch_size * M * N
                out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
                updated_tensors.append(out_tensor)

                a_batch_stride   = M * K
                b_batch_stride   = K * N
                out_batch_stride = M * N

                param_in = make_param([
                    batch_size, M, K, N,
                    a_batch_stride, b_batch_stride, out_batch_stride
                ])
                workgroup = (
                    (M          + LOCAL_X_3D - 1) // LOCAL_X_3D,
                    (N          + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                    (batch_size + LOCAL_Z_3D - 1) // LOCAL_Z_3D
                )
                updated_algorithms.append(self.manager.algorithm(
                    [tensor1, tensor2, out_tensor, param_in],
                    self.shader_matmul,
                    workgroup
                ))

                out_subs  = batch_subs + unshared1 + unshared2
                out_shape = []
                for s in out_subs:
                    if s in subs1:
                        out_shape.append(shape1[subs1.index(s)])
                    else:
                        out_shape.append(shape2[subs2.index(s)])

                return {'tensor': out_tensor, 'shape': out_shape, 'subscripts': out_subs}
            else:
                mult_result = multiply_broadcast(input1, input2)
                keep_subs = set(output_subs)
                mult_result['tensor'], mult_result['shape'], mult_result['subscripts'] = reduce_unused(
                    mult_result['tensor'], mult_result['shape'], mult_result['subscripts'], keep_subs
                )
                return mult_result

        # ============================================================
        # 主逻辑
        # ============================================================

        equation = self.equation.strip().replace(" ", "")
        if "->" in equation:
            inputs_str, output_str = equation.split("->")
        else:
            inputs_str = equation
            output_str = None

        input_specs = inputs_str.split(",")
        num_inputs  = len(input_specs)

        assert num_inputs == len(input_tensors), \
            f"Equation has {num_inputs} inputs but got {len(input_tensors)}"

        input_signatures  = []
        subscript_to_size = {}

        for i, (spec, (tensor, shape)) in enumerate(zip(input_specs, input_tensors)):
            assert len(spec) == len(shape), \
                f"Input {i} spec '{spec}' length {len(spec)} != shape length {len(shape)}"
            subscripts = list(spec)
            input_signatures.append({'tensor': tensor, 'shape': shape, 'subscripts': subscripts})
            for idx, size in zip(subscripts, shape):
                if idx in subscript_to_size:
                    assert subscript_to_size[idx] == size, \
                        f"Subscript '{idx}' has conflicting sizes {subscript_to_size[idx]} and {size}"
                else:
                    subscript_to_size[idx] = size

        if output_str is None:
            all_subs = []
            for sig in input_signatures:
                all_subs.extend(sig['subscripts'])
            seen = set()
            unique_subs = []
            for s in all_subs:
                if s not in seen:
                    unique_subs.append(s); seen.add(s)
            output_subscripts = [s for s in unique_subs if all_subs.count(s) == 1]
        else:
            output_subscripts = list(output_str)

        # 第1步：处理每个输入 - 提取对角线和化简
        processed_inputs = []
        for sig in input_signatures:
            current_tensor = sig['tensor']
            current_shape  = sig['shape']
            current_subs   = sig['subscripts']

            # 为重复下标提取对角线
            seen_cnt = {}
            repeated = []
            for s in current_subs:
                if s in seen_cnt:
                    if s not in repeated:
                        repeated.append(s)
                seen_cnt[s] = seen_cnt.get(s, 0) + 1

            for rep_sub in repeated:
                indices  = [i for i, s in enumerate(current_subs) if s == rep_sub]
                n_repeat = len(indices)

                diag_size = current_shape[indices[0]]
                assert all(current_shape[i] == diag_size for i in indices), \
                    f"Repeated subscript '{rep_sub}' has mismatched dimensions"

                if diag_size == 1:
                    current_shape = [current_shape[i] for i in range(len(current_shape)) if i not in indices[1:]]
                    current_subs  = [current_subs[i]  for i in range(len(current_subs))  if i not in indices[1:]]
                    continue

                if n_repeat == 2 and indices == [indices[0], indices[0] + 1]:
                    outer_size = int(np.prod(current_shape[:indices[0]])) if indices[0] > 0 else 1
                    inner_size = int(np.prod(current_shape[indices[1] + 1:])) if indices[1] + 1 < len(current_shape) else 1

                    out_size = outer_size * diag_size * inner_size
                    out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
                    updated_tensors.append(out_tensor)

                    stride         = diag_size + 1
                    in_outer_stride = diag_size * diag_size * inner_size
                    diag_stride    = stride * inner_size
                    out_row_stride = diag_size * inner_size

                    param_in = make_param([
                        outer_size, diag_size, inner_size,
                        stride, in_outer_stride, diag_stride, out_row_stride
                    ])
                    workgroup = (
                        (outer_size + LOCAL_X_3D - 1) // LOCAL_X_3D,
                        (diag_size  + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
                        (inner_size + LOCAL_Z_3D - 1) // LOCAL_Z_3D
                    )
                    updated_algorithms.append(self.manager.algorithm(
                        [current_tensor, out_tensor, param_in],
                        self.shader_diagonal,
                        workgroup
                    ))

                    current_tensor = out_tensor
                    current_shape  = current_shape[:indices[0]] + [diag_size] + current_shape[indices[1] + 1:]
                    current_subs   = current_subs[:indices[0]]  + [rep_sub]   + current_subs[indices[1] + 1:]
                else:
                    # 通用情况：非连续或多重重复下标
                    first_idx         = indices[0]
                    remaining_indices = indices[1:]
                    kept_dims         = [j for j in range(len(current_shape)) if j not in remaining_indices]
                    new_shape         = [current_shape[j] for j in kept_dims]
                    out_size          = int(np.prod(new_shape))

                    index_map      = [0] * out_size
                    total_elements = int(np.prod(current_shape))
                    multi_idx      = [0] * len(current_shape)

                    for flat_idx in range(total_elements):
                        first_val  = multi_idx[first_idx]
                        all_equal  = all(multi_idx[idx] == first_val for idx in remaining_indices)
                        if all_equal:
                            out_flat = 0; multiplier = 1
                            for ki in range(len(kept_dims) - 1, -1, -1):
                                out_flat  += multi_idx[kept_dims[ki]] * multiplier
                                multiplier *= new_shape[ki]
                            index_map[out_flat] = flat_idx
                        if flat_idx + 1 < total_elements:
                            for ki in range(len(current_shape) - 1, -1, -1):
                                multi_idx[ki] += 1
                                if multi_idx[ki] < current_shape[ki]:
                                    break
                                multi_idx[ki] = 0

                    out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
                    updated_tensors.append(out_tensor)
                    index_map_tensor = self.manager.tensor(np.array(index_map, dtype=np.float32))
                    updated_tensors.append(index_map_tensor)

                    param_in = make_param([out_size])
                    workgroup = ((out_size + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)
                    updated_algorithms.append(self.manager.algorithm(
                        [current_tensor, out_tensor, index_map_tensor, param_in],
                        self.shader_diagonal_general,
                        workgroup
                    ))

                    current_tensor = out_tensor
                    current_shape  = new_shape
                    current_subs   = [current_subs[j] for j in range(len(current_subs)) if j not in remaining_indices]

            # 决定保留哪些下标
            other_subs = set()
            for other_sig in input_signatures:
                if other_sig is not sig:
                    other_subs.update(other_sig['subscripts'])
            other_subs.update(output_subscripts)

            current_tensor, current_shape, current_subs = reduce_unused(
                current_tensor, current_shape, current_subs, other_subs
            )

            processed_inputs.append({
                'tensor': current_tensor,
                'shape': current_shape,
                'subscripts': current_subs
            })

        # 第2步：缩约所有输入
        result = processed_inputs[0]
        for i in range(1, len(processed_inputs)):
            keep_subs_for_this_step = set(output_subscripts)
            for j in range(i + 1, len(processed_inputs)):
                keep_subs_for_this_step.update(processed_inputs[j]['subscripts'])

            result = contract_two(
                result, processed_inputs[i],
                [s for s in keep_subs_for_this_step]
            )

        # 第3步：转置到输出顺序
        result_tensor = result['tensor']
        result_shape  = result['shape']
        result_subs   = result['subscripts']

        if result_subs != output_subscripts:
            perm = [result_subs.index(s) for s in output_subscripts]
            result_tensor, result_shape = transpose_axes(result_tensor, result_shape, perm)

        return [(result_tensor, result_shape)]

