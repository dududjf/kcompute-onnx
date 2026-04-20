import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to


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
        self.shader_mul = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InBuf1 { float in1[]; };
layout(binding = 1) readonly buffer InBuf2 { float in2[]; };
layout(binding = 2) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float size_x_1_f = 0;
layout(constant_id = 1) const float size_y_1_f = 0;
layout(constant_id = 2) const float size_z_1_f = 0;
layout(constant_id = 3) const float size_x_2_f = 0;
layout(constant_id = 4) const float size_y_2_f = 0;
layout(constant_id = 5) const float size_z_2_f = 0;
layout(constant_id = 6) const float stride_x_1_f = 0;
layout(constant_id = 7) const float stride_y_1_f = 0;
layout(constant_id = 8) const float stride_x_2_f = 0;
layout(constant_id = 9) const float stride_y_2_f = 0;
layout(constant_id = 10) const float stride_x_f = 0;
layout(constant_id = 11) const float stride_y_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    
    uint size_x_1 = uint(size_x_1_f);
    uint size_y_1 = uint(size_y_1_f);
    uint size_z_1 = uint(size_z_1_f);
    uint size_x_2 = uint(size_x_2_f);
    uint size_y_2 = uint(size_y_2_f);
    uint size_z_2 = uint(size_z_2_f);
    
    uint stride_x_1 = uint(stride_x_1_f);
    uint stride_y_1 = uint(stride_y_1_f);
    uint stride_x_2 = uint(stride_x_2_f);
    uint stride_y_2 = uint(stride_y_2_f);
    uint stride_x = uint(stride_x_f);
    uint stride_y = uint(stride_y_f);
    
    uint x_1 = min(gx, size_x_1 - 1);
    uint y_1 = min(gy, size_y_1 - 1);
    uint z_1 = min(gz, size_z_1 - 1);
    uint x_2 = min(gx, size_x_2 - 1);
    uint y_2 = min(gy, size_y_2 - 1);
    uint z_2 = min(gz, size_z_2 - 1);
    
    out_data[gx * stride_x + gy * stride_y + gz] = 
        in1[x_1 * stride_x_1 + y_1 * stride_y_1 + z_1] * 
        in2[x_2 * stride_x_2 + y_2 * stride_y_2 + z_2];
}
""")
        # 沿轴进行 ReduceSum 的 Shader
        self.shader_reduce = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InBuf { float in_data[]; };
layout(binding = 1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float outer_size_f = 0;
layout(constant_id = 1) const float reduce_size_f = 0;
layout(constant_id = 2) const float inner_size_f = 0;

void main() {
    uint outer = gl_GlobalInvocationID.x;
    uint inner = gl_GlobalInvocationID.y;
    
    uint reduce_size = uint(reduce_size_f);
    uint inner_size = uint(inner_size_f);
    
    uint base = outer * reduce_size * inner_size + inner;
    
    float sum = 0.0;
    for (uint i = 0; i < reduce_size; ++i) {
        sum += in_data[base];
        base += inner_size;
    }
    
    out_data[outer * inner_size + inner] = sum;
}
""")
        # 全局 ReduceSum 的专用 Shader（将整个张量求和为标量）
        self.shader_reduce_all = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InBuf { float in_data[]; };
layout(binding = 1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float total_size_f = 0;

void main() {
    uint total_size = uint(total_size_f);
    
    float sum = 0.0;
    for (uint i = 0; i < total_size; ++i) {
        sum += in_data[i];
    }
    
    out_data[0] = sum;
}
""")
        # 单轴交换的转置 Shader
        self.shader_transpose = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InBuf { float in_data[]; };
layout(binding = 1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float pre_block_size_f = 0;
layout(constant_id = 1) const float post_block_size_f = 0;
layout(constant_id = 2) const float axis_dimension_f = 0;
layout(constant_id = 3) const float stride_y_f = 0;
layout(constant_id = 4) const float stride_x_f = 0;
layout(constant_id = 5) const float pre_post_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.y;
    uint gy = gl_GlobalInvocationID.x;
    uint gz = gl_GlobalInvocationID.z;
    
    uint post_block_size = uint(post_block_size_f);
    uint stride_y = uint(stride_y_f);
    uint stride_x = uint(stride_x_f);
    uint pre_post = uint(pre_post_f);
    
    uint in_index = gx * stride_x + gy * stride_y + gz * post_block_size;
    uint out_index = gx * stride_x + gz * pre_post + gy * post_block_size;
    
    for (uint i = 0; i < post_block_size; ++i, ++out_index, ++in_index) {
        out_data[out_index] = in_data[in_index];
    }
}
""")
        # 对角线提取的 Shader
        self.shader_diagonal = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InBuf { float in_data[]; };
layout(binding = 1) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float outer_size_f = 0;
layout(constant_id = 1) const float diag_size_f = 0;
layout(constant_id = 2) const float inner_size_f = 0;
layout(constant_id = 3) const float stride_f = 0;
layout(constant_id = 4) const float in_outer_stride_f = 0;
layout(constant_id = 5) const float diag_stride_f = 0;
layout(constant_id = 6) const float out_row_stride_f = 0;

void main() {
    uint outer = gl_GlobalInvocationID.x;
    uint diag_idx = gl_GlobalInvocationID.y;
    uint inner = gl_GlobalInvocationID.z;
    
    uint in_outer_stride = uint(in_outer_stride_f);
    uint diag_stride = uint(diag_stride_f);
    uint out_row_stride = uint(out_row_stride_f);
    
    uint in_base = outer * in_outer_stride + diag_idx * diag_stride + inner;
    uint out_base = outer * out_row_stride + diag_idx * uint(inner_size_f) + inner;
    
    out_data[out_base] = in_data[in_base];
}
""")
        # 分批矩阵乘法的 Shader
        self.shader_matmul = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InBuf1 { float in1[]; };
layout(binding = 1) readonly buffer InBuf2 { float in2[]; };
layout(binding = 2) writeonly buffer OutBuf { float out_data[]; };

layout(constant_id = 0) const float batch_size_f = 0;
layout(constant_id = 1) const float M_f = 0;
layout(constant_id = 2) const float K_f = 0;
layout(constant_id = 3) const float N_f = 0;
layout(constant_id = 4) const float a_batch_stride_f = 0;
layout(constant_id = 5) const float b_batch_stride_f = 0;
layout(constant_id = 6) const float out_batch_stride_f = 0;

void main() {
    uint batch = gl_GlobalInvocationID.z;
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    
    uint K = uint(K_f);
    uint N = uint(N_f);
    uint a_batch_stride = uint(a_batch_stride_f);
    uint b_batch_stride = uint(b_batch_stride_f);
    uint out_batch_stride = uint(out_batch_stride_f);
    
    uint a_base = batch * a_batch_stride + row * K;
    uint b_base = batch * b_batch_stride + col;
    
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += in1[a_base] * in2[b_base];
        ++a_base;
        b_base += N;
    }
    
    out_data[batch * out_batch_stride + row * N + col] = sum;
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"EinsumOp({device_name}, equation='{self.equation}')"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
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

        def reduce_unused(tensor, shape, subscripts, keep_subs):
            """化简（求和）不在 keep_subs 中的轴"""
            reduce_indices = [i for i, s in enumerate(subscripts) if s not in keep_subs]

            if not reduce_indices:
                return tensor, shape, subscripts

            if len(reduce_indices) == len(shape):
                total_size = int(np.prod(shape))
                out_tensor = self.manager.tensor(np.zeros(1, dtype=np.float32))
                updated_tensors.append(out_tensor)

                # 使用专用的全局 reduce shader
                workgroup = (1, 1, 1)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor, out_tensor],
                    self.shader_reduce_all,
                    workgroup,
                    [total_size],
                    []
                ))

                return out_tensor, [], []

            # 逐个化简轴（从最高到最低的索引）
            for idx in sorted(reduce_indices, reverse=True):
                if shape[idx] == 1:
                    # 仅挤压
                    shape = shape[:idx] + shape[idx+1:]
                    subscripts = subscripts[:idx] + subscripts[idx+1:]
                else:
                    # 使用 reduce shader
                    outer_size = int(np.prod(shape[:idx])) if idx > 0 else 1
                    reduce_size = shape[idx]
                    inner_size = int(np.prod(shape[idx+1:])) if idx < len(shape)-1 else 1

                    out_size = outer_size * inner_size
                    out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
                    updated_tensors.append(out_tensor)

                    workgroup = (outer_size, inner_size, 1)
                    updated_algorithms.append(self.manager.algorithm(
                        [tensor, out_tensor],
                        self.shader_reduce,
                        workgroup,
                        [outer_size, reduce_size, inner_size],
                        []
                    ))

                    tensor = out_tensor
                    shape = shape[:idx] + shape[idx+1:]
                    subscripts = subscripts[:idx] + subscripts[idx+1:]

            return tensor, shape, subscripts

        def transpose_axes(tensor, shape, perm):
            """根据排列转置张量"""
            if perm == [i for i in range(len(perm))]:
                return tensor, shape

            out_shape = [shape[i] for i in perm]
            total_size = int(np.prod(out_shape))
            leading_size = 1
            suffix = [i for i in range(len(shape))]

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
                    workgroup = (pre_block_size, leading_size, axis_dimension)

                    # 预计算 stride 常量
                    stride_y = axis_dimension * post_block_size
                    stride_x = pre_block_size * stride_y
                    pre_post = pre_block_size * post_block_size

                    updated_algorithms.append(self.manager.algorithm(
                        [current_tensor, out_tensor],
                        self.shader_transpose,
                        workgroup,
                        [pre_block_size, post_block_size, axis_dimension, stride_y, stride_x, pre_post],
                        []
                    ))

                    current_tensor = out_tensor

                suffix.remove(perm[i])
                leading_size *= out_shape[i]

            return current_tensor, out_shape

        def multiply_broadcast(input1, input2):
            """带广播的逐元素乘法"""
            shape1 = input1['shape']
            shape2 = input2['shape']
            subs1 = input1['subscripts']
            subs2 = input2['subscripts']

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
                    dim1 = shape1[subs1.index(s)]
                    dim2 = shape2[subs2.index(s)]
                    assert dim1 == dim2 or dim1 == 1 or dim2 == 1, \
                        f"Incompatible dimensions for subscript '{s}': {dim1} vs {dim2}"
                    out_shape.append(max(dim1, dim2))
                elif s in subs1:
                    out_shape.append(shape1[subs1.index(s)])
                else:
                    out_shape.append(shape2[subs2.index(s)])

            bcast_shape1 = []
            bcast_shape2 = []

            for s in combined_subs:
                if s in subs1:
                    bcast_shape1.append(shape1[subs1.index(s)])
                else:
                    bcast_shape1.append(1)

                if s in subs2:
                    bcast_shape2.append(shape2[subs2.index(s)])
                else:
                    bcast_shape2.append(1)

            tensor1 = input1['tensor']
            tensor2 = input2['tensor']

            if len(bcast_shape1) > 3:
                batch_bcast1 = bcast_shape1[:-2]
                batch_bcast2 = bcast_shape2[:-2]
                batch_out = out_shape[:-2]

                if batch_bcast1 != batch_out and not all(e == 1 for e in batch_bcast1):
                    final_shape1 = batch_out + bcast_shape1[-2:]
                    tensor1 = broadcast_to(tensor1, bcast_shape1, final_shape1,
                                         updated_algorithms, updated_tensors, self.manager)
                    bcast_shape1 = final_shape1

                if batch_bcast2 != batch_out and not all(e == 1 for e in batch_bcast2):
                    final_shape2 = batch_out + bcast_shape2[-2:]
                    tensor2 = broadcast_to(tensor2, bcast_shape2, final_shape2,
                                         updated_algorithms, updated_tensors, self.manager)
                    bcast_shape2 = final_shape2

            if len(bcast_shape1) == 0:
                bcast_shape1 = [1, 1, 1]
                bcast_shape2 = [1, 1, 1]
                out_shape_3d = (1, 1, 1)
            elif len(bcast_shape1) == 1:
                bcast_shape1 = [bcast_shape1[0], 1, 1]
                bcast_shape2 = [bcast_shape2[0], 1, 1]
                out_shape_3d = (out_shape[0], 1, 1)
            elif len(bcast_shape1) == 2:
                bcast_shape1 = [bcast_shape1[0], bcast_shape1[1], 1]
                bcast_shape2 = [bcast_shape2[0], bcast_shape2[1], 1]
                out_shape_3d = (out_shape[0], out_shape[1], 1)
            elif len(bcast_shape1) == 3:
                out_shape_3d = (out_shape[0], out_shape[1], out_shape[2])
            else:
                batch1 = int(np.prod(bcast_shape1[:-2]))
                bcast_shape1 = [batch1, bcast_shape1[-2], bcast_shape1[-1]]
                batch2 = int(np.prod(bcast_shape2[:-2]))
                bcast_shape2 = [batch2, bcast_shape2[-2], bcast_shape2[-1]]
                batch_out = int(np.prod(out_shape[:-2]))
                out_shape_3d = (int(batch_out), out_shape[-2], out_shape[-1])

            out_size = int(np.prod(out_shape))
            out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
            updated_tensors.append(out_tensor)

            stride_x_1 = bcast_shape1[1] * bcast_shape1[2]
            stride_y_1 = bcast_shape1[2]
            stride_x_2 = bcast_shape2[1] * bcast_shape2[2]
            stride_y_2 = bcast_shape2[2]
            stride_x = max(bcast_shape1[1], bcast_shape2[1]) * max(bcast_shape1[2], bcast_shape2[2])
            stride_y = max(bcast_shape1[2], bcast_shape2[2])

            updated_algorithms.append(self.manager.algorithm(
                [tensor1, tensor2, out_tensor],
                self.shader_mul,
                out_shape_3d,
                bcast_shape1 + bcast_shape2 + [stride_x_1, stride_y_1, stride_x_2, stride_y_2, stride_x, stride_y],
                []
            ))

            return {
                'tensor': out_tensor,
                'shape': out_shape,
                'subscripts': combined_subs
            }

        def contract_two(input1, input2, output_subs):
            """使用 mul 或 matmul 缩约两个输入"""
            subs1 = input1['subscripts']
            subs2 = input2['subscripts']
            subs2_set = set(subs2)
            subs1_set = set(subs1)

            shared = [s for s in subs1 if s in subs2_set]

            if not shared:
                return multiply_broadcast(input1, input2)

            # 检查 matmul 模式
            output_set = set(output_subs)
            contract_dims = [s for s in shared if s not in output_set]
            unshared1 = [s for s in subs1 if s not in subs2_set]
            unshared2 = [s for s in subs2 if s not in subs1_set]

            is_matmul = (len(contract_dims) >= 1 and
                        len(unshared1) >= 1 and len(unshared2) >= 1 and
                        all(s in output_set for s in unshared1) and
                        all(s in output_set for s in unshared2))

            if is_matmul:
                # 使用 matmul shader 进行缩约
                shape1 = input1['shape']
                shape2 = input2['shape']
                batch_subs = [s for s in subs1 if s in subs2 and s in output_set]
                contract_subs = [s for s in shared if s not in output_set]

                batch_size = 1
                for s in batch_subs:
                    idx1 = subs1.index(s)
                    batch_size *= shape1[idx1]

                M = 1
                for s in unshared1:
                    idx1 = subs1.index(s)
                    M *= shape1[idx1]

                K = 1
                for s in contract_subs:
                    idx1 = subs1.index(s)
                    K *= shape1[idx1]

                N = 1
                for s in unshared2:
                    idx2 = subs2.index(s)
                    N *= shape2[idx2]

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

                a_batch_stride = M * K
                b_batch_stride = K * N
                out_batch_stride = M * N

                workgroup = (M, N, batch_size)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor1, tensor2, out_tensor],
                    self.shader_matmul,
                    workgroup,
                    [batch_size, M, K, N, a_batch_stride, b_batch_stride, out_batch_stride],
                    []
                ))

                out_subs = batch_subs + unshared1 + unshared2
                out_shape = []
                for s in out_subs:
                    if s in subs1:
                        idx1 = subs1.index(s)
                        out_shape.append(shape1[idx1])
                    elif s in subs2:
                        idx2 = subs2.index(s)
                        out_shape.append(shape2[idx2])

                return {
                    'tensor': out_tensor,
                    'shape': out_shape,
                    'subscripts': out_subs
                }
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

        # 解析方程
        equation = self.equation.strip().replace(" ", "")
        if "->" in equation:
            inputs_str, output_str = equation.split("->")
        else:
            inputs_str = equation
            output_str = None

        input_specs = inputs_str.split(",")
        num_inputs = len(input_specs)

        assert num_inputs == len(input_tensors), \
            f"Equation has {num_inputs} inputs but got {len(input_tensors)}"

        # 构建特征：解析输入索引和形状
        input_signatures = []
        subscript_to_size = {}

        for i, (spec, (tensor, shape)) in enumerate(zip(input_specs, input_tensors)):
            assert len(spec) == len(shape), \
                f"Input {i} spec '{spec}' length {len(spec)} != shape length {len(shape)}"

            subscripts = [c for c in spec]
            input_signatures.append({
                'tensor': tensor,
                'shape': shape,
                'subscripts': subscripts
            })

            for idx, size in zip(subscripts, shape):
                if idx in subscript_to_size:
                    assert subscript_to_size[idx] == size, \
                        f"Subscript '{idx}' has conflicting sizes {subscript_to_size[idx]} and {size}"
                else:
                    subscript_to_size[idx] = size

        # 确定输出下标
        if output_str is None:
            all_subscripts = []
            for sig in input_signatures:
                all_subscripts.extend(sig['subscripts'])
            unique_subs = []
            seen = set()
            for s in all_subscripts:
                if s not in seen:
                    unique_subs.append(s)
                    seen.add(s)
            output_subscripts = [s for s in unique_subs if all_subscripts.count(s) == 1]
        else:
            output_subscripts = [c for c in output_str]

        output_shape = [subscript_to_size[s] for s in output_subscripts]

        # 处理真正为空的输出（零元素，不是标量）
        if len(output_shape) > 0 and np.prod(output_shape) == 0:
            # 空张量
            result_size = int(np.prod(output_shape))
            result_tensor = self.manager.tensor(np.zeros(max(1, result_size), dtype=np.float32))
            updated_tensors.append(result_tensor)
            return [(result_tensor, output_shape)]

        # 对于标量输出（len(output_shape) == 0），继续正常处理

        # 第1步：处理每个输入 - 提取对角线和化简
        processed_inputs = []
        for sig in input_signatures:
            current_tensor = sig['tensor']
            current_shape = sig['shape']
            current_subs = sig['subscripts']

            # 为重复下标提取对角线
            seen = {}
            repeated = []
            for s in current_subs:
                if s in seen:
                    if s not in repeated:
                        repeated.append(s)
                seen[s] = seen.get(s, 0) + 1

            for rep_sub in repeated:
                indices = [i for i, s in enumerate(current_subs) if s == rep_sub]
                n_repeat = len(indices)

                if n_repeat < 2:
                    continue

                diag_size = current_shape[indices[0]]
                assert all(current_shape[i] == diag_size for i in indices), \
                    f"Repeated subscript '{rep_sub}' has mismatched dimensions"

                if diag_size == 1:
                    # 仅移除多余维度
                    new_shape = [current_shape[i] for i in range(len(current_shape)) if i not in indices[1:]]
                    new_subs = [current_subs[i] for i in range(len(current_subs)) if i not in indices[1:]]
                    current_tensor, current_shape, current_subs = current_tensor, new_shape, new_subs
                    continue

                if n_repeat == 2 and indices == [indices[0], indices[0]+1]:
                    # 连续维度 - 可以使用简化的 shader
                    outer_size = int(np.prod(current_shape[:indices[0]])) if indices[0] > 0 else 1
                    inner_size = int(np.prod(current_shape[indices[1]+1:])) if indices[1]+1 < len(current_shape) else 1

                    out_size = outer_size * diag_size * inner_size
                    out_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
                    updated_tensors.append(out_tensor)

                    stride = diag_size + 1
                    in_outer_stride = diag_size * diag_size * inner_size
                    diag_stride = stride * inner_size
                    out_row_stride = diag_size * inner_size

                    workgroup = (outer_size, diag_size, inner_size)
                    updated_algorithms.append(self.manager.algorithm(
                        [current_tensor, out_tensor],
                        self.shader_diagonal,
                        workgroup,
                        [outer_size, diag_size, inner_size, stride, in_outer_stride, diag_stride, out_row_stride],
                        []
                    ))

                    new_shape = current_shape[:indices[0]] + [diag_size] + current_shape[indices[1]+1:]
                    new_subs = current_subs[:indices[0]] + [rep_sub] + current_subs[indices[1]+1:]
                    current_tensor, current_shape, current_subs = out_tensor, new_shape, new_subs
                else:
                    # 通用情况：处理非连续或多重复重复下标
                    first_idx = indices[0]
                    remaining_indices = indices[1:]

                    new_shape = [current_shape[i] for i in range(len(current_shape)) if i not in remaining_indices]
                    out_size = int(np.prod(new_shape))

                    old_tensor_data = current_tensor.data()
                    out_tensor_data = np.zeros(out_size, dtype=np.float32)

                    total_elements = int(np.prod(current_shape))

                    for flat_idx in range(total_elements):
                        multi_idx = []
                        temp = flat_idx
                        for i in range(len(current_shape)-1, -1, -1):
                            multi_idx.insert(0, temp % current_shape[i])
                            temp //= current_shape[i]

                        first_val = multi_idx[first_idx]
                        all_equal = all(multi_idx[idx] == first_val for idx in remaining_indices)

                        if all_equal:
                            out_multi_idx = [multi_idx[i] for i in range(len(current_shape)) if i not in remaining_indices]
                            out_flat_idx = 0
                            multiplier = 1
                            for i in range(len(out_multi_idx)-1, -1, -1):
                                out_flat_idx += out_multi_idx[i] * multiplier
                                multiplier *= new_shape[i]

                            out_tensor_data[out_flat_idx] = old_tensor_data[flat_idx]

                    out_tensor = self.manager.tensor(out_tensor_data)
                    updated_tensors.append(out_tensor)

                    new_subs = [current_subs[i] for i in range(len(current_subs)) if i not in remaining_indices]
                    current_tensor, current_shape, current_subs = out_tensor, new_shape, new_subs

            # 决定保留哪些下标
            other_subs = set()
            for other_sig in input_signatures:
                if other_sig is not sig:
                    other_subs.update(other_sig['subscripts'])
            other_subs.update(output_subscripts)

            # Reduce 没有使用到的维度
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
            # 对于中间的缩约，保持以下维度：
            # 1. 在最终输出中的维度
            # 2. 在任何剩余输入中的维度（尚未缩约的）
            keep_subs_for_this_step = set(output_subscripts)
            for j in range(i+1, len(processed_inputs)):
                keep_subs_for_this_step.update(processed_inputs[j]['subscripts'])

            result = contract_two(
                result, processed_inputs[i], [s for s in keep_subs_for_this_step]
            )

        # 第3步：转置到输出顺序
        result_tensor = result['tensor']
        result_shape = result['shape']
        result_subs = result['subscripts']

        if result_subs != output_subscripts:
            perm = [result_subs.index(s) for s in output_subscripts]
            result_tensor, result_shape = transpose_axes(result_tensor, result_shape, perm)

        return [(result_tensor, result_shape)]
