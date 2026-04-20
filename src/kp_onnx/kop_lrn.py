import numpy as np
import kp
from .shader_utils import compile_source


class LRNOp:
    def __init__(self, manager: kp.Manager, alpha: float = 0.0001, beta: float = 0.75, bias: float = 1.0, size: int = 5):
        self.manager = manager
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.size = size
        self.compiled_shader = compile_source('''
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(binding=0) buffer buf_in_tensor    { float in_tensor[]; };
layout(binding=1) buffer buf_out_tensor   { float out_tensor[]; };

layout(constant_id=0) const float batch_dim_f      = 0;
layout(constant_id=1) const float channel_dim_f    = 0; 
layout(constant_id=2) const float spatial_dim_f    = 0;
layout(constant_id=3) const float alpha_div_size_f = 0;
layout(constant_id=4) const float beta_f           = 0;
layout(constant_id=5) const float bias_f           = 0;
layout(constant_id=6) const float pad_before_f     = 0;
layout(constant_id=7) const float pad_after_f      = 0;

void main() {
    uint spatial_idx = gl_GlobalInvocationID.x; 
    uint channel_idx = gl_GlobalInvocationID.y; 
    uint batch_idx   = gl_GlobalInvocationID.z; 

    uint channel_dim = uint(channel_dim_f); 
    uint spatial_dim = uint(spatial_dim_f);

    uint pad_before  = uint(pad_before_f);
    uint pad_after   = uint(pad_after_f);

    uint begin = 0;
    if (channel_idx >= pad_before) {
        begin = channel_idx - pad_before;
    }

    uint end = channel_idx + pad_after;
    if (end > channel_dim) {
        end = channel_dim;
    }

    uint batch_pitch = channel_dim * spatial_dim;
    uint base_offset = batch_idx * batch_pitch + spatial_idx;

    float sum_sq = 0.0;
    uint read_idx = base_offset + begin * spatial_dim;

    for (uint k = begin; k < end; ++k) {
        float val = in_tensor[read_idx];
        sum_sq += val * val;
        read_idx += spatial_dim;
    }

    uint curr_idx = base_offset + channel_idx * spatial_dim;
    float val_x = in_tensor[curr_idx];

    float denom_base = bias_f + alpha_div_size_f * sum_sq;
    float denom = pow(denom_base, beta_f);

    out_tensor[curr_idx] = val_x / denom;
}
''')

    def __repr__(self):
        return f"LRNOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"LRNOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

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
        tensor_in, shape_in = input_tensors[0]

        assert len(shape_in) == 4, f"LRN only applies on 4D tensors but got shape={shape_in}"
        assert self.size > 0, "LRNOp.size must be > 0"

        batch_size = shape_in[0]
        channel_size = shape_in[1]
        spatial_size = np.prod(shape_in[2:])
        total_size = np.prod(shape_in)

        tensor_out = self.manager.tensor(np.zeros((total_size,), dtype=np.float32))
        updated_tensors.append(tensor_out)

        pad_before = (self.size - 1) // 2
        pad_after = (self.size // 2) + 1
        alpha_div_size = self.alpha / self.size

        params = [
            batch_size,
            channel_size,
            spatial_size,
            alpha_div_size,
            self.beta,
            self.bias,
            pad_before,
            pad_after
        ]

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            self.compiled_shader,
            (spatial_size, channel_size, batch_size),
            params,
            []
        ))

        return [(tensor_out, shape_in)]
