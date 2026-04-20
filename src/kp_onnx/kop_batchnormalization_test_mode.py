import kp
import numpy as np
from .shader_utils import compile_source


class BatchNormalizationTestModeOp:
    def __init__(self, manager: kp.Manager, epsilon=1e-05, momentum=0.9, training_mode=0):
        self.manager = manager
        self.epsilon = epsilon
        self.momentum = momentum
        self.training_mode = training_mode
        self.compiled_shader_test = compile_source("""
#version 450

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) readonly buffer buf_in_tensor { float in_tensor[]; };
layout (binding = 1) readonly buffer buf_scale_tensor { float scale_tensor[]; };
layout (binding = 2) readonly buffer buf_mean_tensor { float mean_tensor[]; };
layout (binding = 3) readonly buffer buf_var_tensor { float var_tensor[]; };
layout (binding = 4) readonly buffer buf_bias_tensor { float bias_tensor[]; };
layout (binding = 5) writeonly buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float stride_x_f = 0;
layout (constant_id = 1) const float stride_y_f = 0;
layout (constant_id = 2) const float epsilon_f = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    uint stride_x = uint(stride_x_f);
    uint stride_y = uint(stride_y_f);
    
    uint p = gx * stride_x + gy * stride_y + gz;

    float scale = scale_tensor[gy];
    float mean  = mean_tensor[gy];
    float var   = var_tensor[gy];
    float bias  = bias_tensor[gy];
    float x = in_tensor[p];
    out_tensor[p] = scale * (x - mean) / sqrt(var + epsilon_f) + bias;
}
""")

    def __repr__(self):
        return f"BatchNormalizationTestModeOp({self.manager.get_device_properties()['device_name']})"

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

        workgroup = (size_x, size_y, size_z)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_scale, tensor_saved_mean, tensor_saved_var, tensor_bias, tensor_out],
            self.compiled_shader_test,
            workgroup,
            [size_y * size_z, size_z, self.epsilon],
        ))
        updated_tensors.append(tensor_out)

        return [(tensor_out, shape_in)]
