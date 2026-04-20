import numpy as np
import kp
from .shader_utils import compile_source, broadcast_to, LOCAL_X_1D, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class RmsNormalizationOp:
    def __init__(self, manager: kp.Manager, axis: int = -1, epsilon: float = 1e-05, stash_type: int = 1):
        self.manager = manager
        self.axis = axis
        self.epsilon = epsilon
        self.stash_type = stash_type

        self.compiled_shader_power = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];  }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf  {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint total_size = params[0];
    if (gx >= total_size) return;

    out_tensor[gx] = pow(in_tensor[gx], 2);
}}
""")

        self.compiled_shader_mean = compile_source(f"""
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
    out_tensor[out_offset] = acc / float(dimension);
}}
""")

        self.compiled_shader_multiply = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf1  {{ float in_tensor_1[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer InBuf2  {{ float in_tensor_2[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf  {{ float out_tensor[];  }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint max_x = params[0];
    uint max_y = params[1];
    uint max_z = params[2];
    if (gx >= max_x || gy >= max_y || gz >= max_z) return;

    uint size_x_in = params[3];
    uint size_y_in = params[4];
    uint size_z_in = params[5];
    uint size_x_scale = params[6];
    uint size_y_scale = params[7];
    uint size_z_scale = params[8];

    uint stride_y_in = size_z_in;
    uint stride_x_in = size_y_in * stride_y_in;
    uint stride_y_scale = size_z_scale;
    uint stride_x_scale = size_y_scale * stride_y_scale;
    uint stride_y = size_z_in;
    uint stride_x = size_y_in * stride_y;

    uint x_in = min(gx, size_x_in - 1);
    uint y_in = min(gy, size_y_in - 1);
    uint z_in = min(gz, size_z_in - 1);
    uint x_scale = min(gx, size_x_scale - 1);
    uint y_scale = min(gy, size_y_scale - 1);
    uint z_scale = min(gz, size_z_scale - 1);

    uint p_in = x_in * stride_x_in + y_in * stride_y_in + z_in;
    uint p_scale = x_scale * stride_x_scale + y_scale * stride_y_scale + z_scale;

    out_tensor[gx * stride_x + gy * stride_y + gz] = in_tensor_1[p_in] * in_tensor_2[p_scale];
}}
""")

        self.compiled_shader_apply_kept = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_1D}) in;

layout (std430, set = 0, binding = 0) readonly  buffer InBuf   {{ float in_tensor[];   }};
layout (std430, set = 0, binding = 1) readonly  buffer MeanBuf {{ float mean_tensor[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBuf  {{ float out_tensor[];  }};
layout (std430, set = 0, binding = 3) readonly  buffer UIParams {{ uint params[]; }};

void main() {{
    uint kept_idx = gl_GlobalInvocationID.x;
    uint kept_size = params[0];
    if (kept_idx >= kept_size) return;

    uint reduce_size = params[1];
    float epsilon = uintBitsToFloat(params[2]);

    float inv_rms = 1.0 / sqrt(mean_tensor[kept_idx] + epsilon);
    uint base = kept_idx * reduce_size;
    for (uint s = 0; s < reduce_size; ++s, ++base) {{
        out_tensor[base] = in_tensor[base] * inv_rms;
    }}
}}
""")

    def __repr__(self):
        return f"RmsNormalizationOp({self.manager.get_device_properties()['device_name']})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_in = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(shape_in)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert self.stash_type == 1, "RMSNormalization not implemented for stash_type != 1."
        tensor_in, shape_in = input_tensors[0]
        ori_tensor = tensor_in
        tensor_scale, shape_scale = input_tensors[1]

        axis = self.axis if self.axis >= 0 else len(shape_in) + self.axis

        axes = list(range(axis, len(shape_in)))
        axis_present = [False] * len(shape_in)
        for i in axes:
            axis_present[i] = True

        # Power shader
        total_size = int(np.prod(shape_in))
        tensor_pow = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
        updated_tensors.append(tensor_pow)

        params_pow = np.array([total_size], dtype=np.uint32)
        param_pow_in = self.manager.tensor_t(params_pow, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_pow_in])).eval()

        workgroup_pow = ((total_size + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_pow, param_pow_in],
            self.compiled_shader_power,
            workgroup_pow,
        ))

        # Mean reduction
        tensor_mean = tensor_pow
        block_size = 1
        for i in reversed(range(len(shape_in))):
            if axis_present[i] and shape_in[i] > 1:
                group_x = int(np.prod(shape_in[:i])) if i >= 0 else 1
                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                tensor_in_mean = tensor_mean
                tensor_mean = self.manager.tensor(numpy_out)

                params = np.array([group_x, block_size, shape_in[i]], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

                workgroup = (
                    (group_x + LOCAL_X_2D - 1) // LOCAL_X_2D,
                    (block_size + LOCAL_Y_2D - 1) // LOCAL_Y_2D,
                    1,
                )

                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in_mean, tensor_mean, param_in],
                    self.compiled_shader_mean,
                    workgroup,
                ))
                updated_tensors.append(tensor_mean)
            else:
                block_size *= int(shape_in[i])

        # Apply inv_rms
        kept_size = int(np.prod(shape_in[:axis])) if axis > 0 else 1
        reduce_size = int(np.prod(shape_in[axis:])) if axis < len(shape_in) else 1
        tensor_mat = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
        updated_tensors.append(tensor_mat)

        epsilon_uint = np.float32(self.epsilon).view(np.uint32)
        params_apply = np.array([kept_size, reduce_size, epsilon_uint], dtype=np.uint32)
        param_apply_in = self.manager.tensor_t(params_apply, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_apply_in])).eval()

        workgroup_apply = ((kept_size + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)
        updated_algorithms.append(self.manager.algorithm(
            [ori_tensor, tensor_mean, tensor_mat, param_apply_in],
            self.compiled_shader_apply_kept,
            workgroup_apply,
        ))

        # Multiply with scale
        new_shape_scale = [1] * (len(shape_in) - len(shape_scale)) + shape_scale

        new_tensor_scale = tensor_scale
        algorithms, next_tensors = [], []
        if shape_in[:-2] != new_shape_scale[:-2] and not all(e == 1 for e in new_shape_scale[:-2]):
            final_shape_scale = shape_in[:-2] + list(new_shape_scale[-2:])
            new_tensor_scale = broadcast_to(tensor_scale, new_shape_scale, final_shape_scale,
                                            algorithms, next_tensors, self.manager)
            updated_algorithms.extend(algorithms)
            new_shape_scale = final_shape_scale

        if len(shape_in) == 1:
            size_x_in, size_y_in, size_z_in = shape_in[0], 1, 1
            size_x_scale, size_y_scale, size_z_scale = new_shape_scale[0], 1, 1
        elif len(shape_in) == 2:
            size_x_in, size_y_in, size_z_in = shape_in[0], shape_in[1], 1
            size_x_scale, size_y_scale, size_z_scale = new_shape_scale[0], new_shape_scale[1], 1
        else:
            size_x_in, size_y_in, size_z_in = int(np.prod(shape_in[:-2])), shape_in[-2], shape_in[-1]
            size_x_scale, size_y_scale, size_z_scale = \
                int(np.prod(new_shape_scale[:-2])), new_shape_scale[-2], new_shape_scale[-1]

        tensor_out = self.manager.tensor(np.zeros(total_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        params_mul = np.array([size_x_in, size_y_in, size_z_in,
                               size_x_in, size_y_in, size_z_in,
                               size_x_scale, size_y_scale, size_z_scale], dtype=np.uint32)
        param_mul_in = self.manager.tensor_t(params_mul, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_mul_in])).eval()

        workgroup_mul = (
            (size_x_in + LOCAL_X_3D - 1) // LOCAL_X_3D,
            (size_y_in + LOCAL_Y_3D - 1) // LOCAL_Y_3D,
            (size_z_in + LOCAL_Z_3D - 1) // LOCAL_Z_3D,
        )

        updated_algorithms.append(self.manager.algorithm(
            [tensor_mat, new_tensor_scale, tensor_out, param_mul_in],
            self.compiled_shader_multiply,
            workgroup_mul,
        ))

        return [(tensor_out, shape_in)]
