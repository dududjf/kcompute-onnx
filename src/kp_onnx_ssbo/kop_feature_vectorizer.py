import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D


class FeatureVectorizerOp:
    def __init__(self, manager: kp.Manager, inputdimensions: list[int] = []):
        self.manager = manager
        self.inputdimensions = inputdimensions
        self.compile_shader = compile_source(f"""
#version 450

layout(local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;
layout(std430, set = 0, binding = 0) readonly  buffer InBuf    {{ float in_tensor[];  }};
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf   {{ float out_tensor[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer UIParams {{ uint params[]; }};

void main() 
{{
    uint batch = params[0];
    uint cut_width = params[1];
    uint input_width = params[2];
    uint out_axis_offset = params[3];
    uint out_axis_dim = params[4];
    
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    if(gx >= batch || gy >= cut_width) return;
    
    uint in_offset = gx * input_width + gy;
    uint out_offset = gx * out_axis_dim + out_axis_offset + gy;
    
    out_tensor[out_offset] = gy < input_width ? in_tensor[in_offset] : 0.0;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"FeatureVectorizerOp({device_name})"

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
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()

        if tensor_out is not None:
            output = tensor_out.data().reshape(output_shape)
        else:
            output = np.array([], dtype=np.float32)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        batch = input_tensors[0][1][0]
        input_widths = []
        output_widths = []

        for (_, shape), cut in zip(input_tensors, self.inputdimensions):
            assert len(shape) == 1 or len(shape) == 2, f"Every input must have 1 or 2 dimensions not {shape}."
            if len(shape) == 1:
                n_cols = 1
            else:
                n_cols = shape[1]
            input_widths.append(n_cols)

            if cut < 0:
                if cut + n_cols > 0:
                    cut += n_cols
                else:
                    cut = 0
            output_widths.append(cut)

        shape_out = [batch, sum(output_widths)]

        if shape_out[1] == 0:
            return [(None, [])]

        else:
            tensor_out = self.manager.tensor(np.zeros(int(np.prod(shape_out)), dtype=np.float32))
            updated_tensors.append(tensor_out)

            offset = 0
            for (tensor, shape), input_width, cut_width in zip(input_tensors, input_widths, output_widths):
                if cut_width == 0:
                    continue
                workgroup = ((batch + LOCAL_X_2D - 1) // LOCAL_X_2D, (cut_width + LOCAL_Y_2D - 1) // LOCAL_Y_2D, 1)
                params = np.array([batch, cut_width, input_width, offset, shape_out[1]], dtype=np.uint32)
                param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
                self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()
                alg = self.manager.algorithm([tensor, tensor_out, param_in], self.compile_shader, workgroup)
                updated_algorithms.append(alg)
                offset += cut_width

            return [(tensor_out, shape_out)]
