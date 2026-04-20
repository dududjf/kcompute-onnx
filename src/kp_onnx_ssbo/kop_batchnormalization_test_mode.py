import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class BatchNormalizationTestModeOp:
    def __init__(self, manager: kp.Manager, epsilon=1e-05, momentum=0.9, training_mode=0):
        self.manager = manager
        self.epsilon = epsilon
        self.momentum = momentum
        self.training_mode = training_mode
        self.compiled_shader_test = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf    {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer ScaleBuf {{ float scale_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer MeanBuf  {{ float mean_tensor[]; }};
layout (std430, set = 0, binding = 3) readonly  buffer VarBuf   {{ float var_tensor[]; }};
layout (std430, set = 0, binding = 4) readonly  buffer BiasBuf  {{ float bias_tensor[]; }};
layout (std430, set = 0, binding = 5) writeonly buffer OutBuf   {{ float out_tensor[]; }};
layout (std430, set = 0, binding = 6) readonly  buffer UIParams {{ uint params[]; }};

void main()
{{
    uint size_x = params[0], size_y = params[1], size_z = params[2];
    uint stride_x = params[3], stride_y = params[4];
    float epsilon = uintBitsToFloat(params[5]);

    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    if (gx >= size_x || gy >= size_y || gz >= size_z) return;

    uint p = gx * stride_x + gy * stride_y + gz;

    float scale = scale_tensor[gy];
    float mean  = mean_tensor[gy];
    float var   = var_tensor[gy];
    float bias  = bias_tensor[gy];
    float x = in_tensor[p];
    out_tensor[p] = scale * (x - mean) / sqrt(var + epsilon) + bias;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BatchNormalizationTestModeOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
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
        tensor_in, shape_in = input_tensors[0]
        tensor_scale, shape_scale = input_tensors[1]
        tensor_bias, shape_bias = input_tensors[2]
        tensor_saved_mean, shape_saved_mean = input_tensors[3]
        tensor_saved_var, shape_saved_var = input_tensors[4]

        assert len(shape_in) >= 2, "Input tensor x must have at least 2 dimensions"

        C = shape_in[1] if len(shape_in) > 1 else 1
        tensor_out = self.manager.tensor(np.zeros(shape_in, dtype=np.float32))

        if len(shape_in) == 2:
            size_x = shape_in[0]
            size_y = C
            size_z = 1
        else:
            size_x = shape_in[0]
            size_y = C
            size_z = int(np.prod(shape_in[2:]))

        stride_x = size_y * size_z
        stride_y = size_z
        epsilon_bits = np.float32(self.epsilon).view(np.uint32)

        params = np.array([size_x, size_y, size_z, stride_x, stride_y, epsilon_bits], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        group_x = (size_x + LOCAL_X_3D - 1) // LOCAL_X_3D
        group_y = (size_y + LOCAL_Y_3D - 1) // LOCAL_Y_3D
        group_z = (size_z + LOCAL_Z_3D - 1) // LOCAL_Z_3D
        workgroup = (group_x, group_y, group_z)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_scale, tensor_saved_mean, tensor_saved_var, tensor_bias, tensor_out, param_in],
            self.compiled_shader_test,
            workgroup
        ))
        updated_tensors.append(tensor_out)

        return [(tensor_out, shape_in)]
