import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D


class BatchNormalizationTrainingModeOp:
    def __init__(self, manager: kp.Manager, epsilon=1e-05, momentum=0.9, training_mode=0):
        self.manager = manager
        self.epsilon = epsilon
        self.momentum = momentum
        self.training_mode = training_mode
        self.compiled_shader_mean = compile_source(f"""
#version 450

layout (local_size_x = {LOCAL_X_1D}) in;
layout (std430, set = 0, binding = 0) readonly  buffer InBuf       {{ float in_tensor[]; }};
layout (std430, set = 0, binding = 1) readonly  buffer MeanBuf     {{ float mean_tensor[]; }};
layout (std430, set = 0, binding = 2) readonly  buffer VarBuf      {{ float var_tensor[];  }};
layout (std430, set = 0, binding = 3) readonly  buffer ScaleBuf    {{ float scale_tensor[]; }};
layout (std430, set = 0, binding = 4) readonly  buffer BiasBuf     {{ float bias_tensor[];  }};
layout (std430, set = 0, binding = 5) writeonly buffer OutMeanBuf  {{ float out_mean_tensor[]; }};
layout (std430, set = 0, binding = 6) writeonly buffer OutVarBuf   {{ float out_var_tensor[];  }};
layout (std430, set = 0, binding = 7) writeonly buffer OutBuf      {{ float out_tensor[];   }};
layout (std430, set = 0, binding = 8) readonly  buffer UIParams    {{ uint params[]; }};

void main()
{{
    uint C = params[0], leading = params[1], trailing = params[2], dt = params[3];
    float momentum = uintBitsToFloat(params[4]);
    float epsilon = uintBitsToFloat(params[5]);

    uint gx = gl_GlobalInvocationID.x;
    if (gx >= C) return;

    float sum = 0.0;
    float sumsq = 0.0;
    uint N = leading * trailing;

    uint base = gx * trailing;
    for (uint ld = 0; ld < leading; ++ld, base += dt) {{
        uint p = base;
        for (uint tr = 0; tr < trailing; ++tr, ++p) {{
            sum += in_tensor[p];
            sumsq += in_tensor[p] * in_tensor[p];
        }}
    }}

    float saved_mean = sum / N;
    float saved_var  = max(0.0, sumsq / N - saved_mean * saved_mean);

    out_mean_tensor[gx] = mean_tensor[gx] * momentum + saved_mean * (1.0 - momentum);
    out_var_tensor[gx]  = var_tensor[gx]  * momentum + saved_var  * (1.0 - momentum);

    base = gx * trailing;
    float s = scale_tensor[gx];
    float b = bias_tensor[gx];
    float denom = sqrt(saved_var + epsilon);
    for (uint ld = 0; ld < leading; ++ld, base += dt) {{
        uint p = base;
        for (uint tr = 0; tr < trailing; ++tr, ++p) {{
            float x = in_tensor[p];
            out_tensor[p] = s * (x - saved_mean) / denom + b;
        }}
    }}
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BatchNormalizationTrainingModeOp({device_name})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([t[0] for t in output_tensor_and_shape]))
        seq.eval()

        output_list = []
        for tensor_out, shape_out in output_tensor_and_shape:
            output_list.append(tensor_out.data().reshape(shape_out))

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return output_list

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert self.training_mode, "BatchNormalizationTrainingModeOp only supports training_mode=1"

        tensor_in, shape_in = input_tensors[0]
        tensor_scale, shape_scale = input_tensors[1]
        tensor_bias, shape_bias = input_tensors[2]
        tensor_mean, shape_mean = input_tensors[3]
        tensor_var, shape_var = input_tensors[4]

        output_tensor_and_shape = []

        C = shape_in[1] if len(shape_in) > 1 else 1
        leading = int(np.prod(shape_in[:1]))
        trailing = int(np.prod(shape_in[2:])) if len(shape_in) > 2 else 1
        dt = C * trailing

        tensor_output_mean = self.manager.tensor(np.zeros(C, dtype=np.float32))
        tensor_output_var = self.manager.tensor(np.zeros(C, dtype=np.float32))
        tensor_out = self.manager.tensor(np.zeros(np.prod(shape_in), dtype=np.float32))

        momentum_bits = np.float32(self.momentum).view(np.uint32)
        epsilon_bits = np.float32(self.epsilon).view(np.uint32)
        params = np.array([C, leading, trailing, dt, momentum_bits, epsilon_bits], dtype=np.uint32)
        param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

        group_x = (C + LOCAL_X_1D - 1) // LOCAL_X_1D
        workgroup = (group_x, 1, 1)

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_mean, tensor_var, tensor_scale, tensor_bias,
             tensor_output_mean, tensor_output_var, tensor_out, param_in],
            self.compiled_shader_mean,
            workgroup
        ))
        updated_tensors.extend([tensor_output_mean, tensor_output_var, tensor_out])

        output_tensor_and_shape.append((tensor_out, shape_in))
        output_tensor_and_shape.append((tensor_output_mean, [C]))
        output_tensor_and_shape.append((tensor_output_var, [C]))
        return output_tensor_and_shape
