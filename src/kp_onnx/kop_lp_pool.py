import kp
import numpy as np
from .shader_utils import compile_source


class LpPoolOp:

    def __init__(self, manager: kp.Manager, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=None,
                 kernel_shape=None, p=2, pads=None, strides=None):
        self.auto_pad = auto_pad
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dilations = dilations
        self.kernel_shape = kernel_shape
        self.p = p
        self.pads = pads
        self.strides = strides
        self.manager = manager
        self.pool_shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout(constant_id = 0)  const float post_size_f = 0.0;
layout(constant_id = 1)  const float input_dim_f = 0.0;
layout(constant_id = 2)  const float output_dim_f = 0.0;
layout(constant_id = 3)  const float kernel_size_f = 0.0;
layout(constant_id = 4)  const float stride_f = 0.0;
layout(constant_id = 5)  const float pad_begin_f = 0.0;
layout(constant_id = 6)  const float dilation_f = 0.0;
layout(constant_id = 7)  const float cip_f = 0.0;
layout(constant_id = 8)  const float p_f = 2.0;
layout(constant_id = 9)  const float inv_p_f = 0.5;

void main() {
    uint pre_idx = gl_GlobalInvocationID.x;
    uint out_dim_idx = gl_GlobalInvocationID.y;
    uint post_idx = gl_GlobalInvocationID.z;

    uint post_size = uint(post_size_f);
    uint input_dim = uint(input_dim_f);
    uint output_dim = uint(output_dim_f);
    uint stride = uint(stride_f);
    uint pad_begin = uint(pad_begin_f);
    uint dilation = uint(dilation_f);
    uint cip = uint(cip_f);
    
    float p = p_f;
    float inv_p = inv_p_f;

    uint start_pos = out_dim_idx * stride - pad_begin;
    float sum_p = 0.0;
    uint valid_count = 0;

    uint base_offset = pre_idx * input_dim * post_size + post_idx;
    uint in_idx_step = post_size * dilation;
    uint in_idx = base_offset + start_pos * post_size;
    
    for (uint k = 0; k < kernel_size_f; ++k) {
        if (start_pos >= 0 && start_pos < input_dim) {
            float val = in_buf[in_idx];
            sum_p += pow(abs(val), p);
            valid_count++;
        } else if (cip == 1) {
            valid_count += 1;
        }
        start_pos += dilation;
        in_idx += in_idx_step;
    }
    
    float denom = (cip == 1) ? kernel_size_f : float(max(valid_count, 1));
    float avg_p = sum_p / denom;
    float result = pow(avg_p * denom, inv_p);
    
    uint out_idx = pre_idx * output_dim * post_size + out_dim_idx * post_size + post_idx;
    out_buf[out_idx] = result;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"LpPoolOp({dev})"

    __str__ = __repr__

    def _get_output_shape_explicit_padding(self, input_spatial_shape, kernel_spatial_shape, strides_spatial, pads,
                                           dilations, ceil_mode):
        dims = len(input_spatial_shape)
        dilations = dilations or [1] * dims
        out_shape = [0] * dims
        pad_list = list(pads or [0] * (2 * dims))

        for dim in range(dims):
            num = (input_spatial_shape[dim] + pad_list[dim] + pad_list[dims + dim]
                   - dilations[dim] * (kernel_spatial_shape[dim] - 1) - 1) / strides_spatial[dim] + 1.0
            if ceil_mode:
                out = int(np.ceil(num))
                if (out - 1) * strides_spatial[dim] >= input_spatial_shape[dim] + pad_list[dim]:
                    out -= 1
            else:
                out = int(np.floor(num))
            out_shape[dim] = out

        pads_new = pad_list[:]
        for dim in range(dims):
            sw = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
            actual = (out_shape[dim] - 1) * strides_spatial[dim] + sw
            extra = actual - input_spatial_shape[dim] - pad_list[dim] - pad_list[dims + dim]
            if extra > 0:
                pads_new[dim] += extra // 2
                pads_new[dims + dim] += extra - extra // 2
        return out_shape, pads_new

    def _get_output_shape_auto_pad(self, auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, dilations):
        out_shape = [0] * len(input_spatial_shape)

        for i in range(len(input_spatial_shape)):
            if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
                out_shape[i] = int(np.floor((input_spatial_shape[i] - 1) / strides_spatial[i])) + 1
            elif auto_pad == "VALID":
                effective_kernel = (kernel_spatial_shape[i] - 1) * dilations[i] + 1
                out_shape[i] = int(np.floor((input_spatial_shape[i] - effective_kernel) / strides_spatial[i])) + 1
        return out_shape

    def _get_pad_shape(self, auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, out_shape):
        pad_shape = [0] * len(input_spatial_shape)
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            for i in range(len(input_spatial_shape)):
                pad_shape[i] = (out_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
        return pad_shape

    def _get_pad_with_auto_pad(self, auto_pad, pad_shape):
        dims = len(pad_shape)
        if auto_pad == "SAME_UPPER":
            return [pad_shape[i] // 2 for i in range(dims)] + [pad_shape[i] - pad_shape[i] // 2 for i in range(dims)]
        elif auto_pad == "SAME_LOWER":
            return [pad_shape[i] - pad_shape[i] // 2 for i in range(dims)] + [pad_shape[i] // 2 for i in range(dims)]
        else:
            return [0] * dims * 2

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

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

        assert len(shape_in) >= 3, f"LpPool expects at least 3D input (N, C, spatial...), got {shape_in}"
        n, c = shape_in[0], shape_in[1]
        input_spatial_shape = shape_in[2:]
        spatial_dims = len(input_spatial_shape)

        # ONNX规范：kernel_shape必须提供
        assert self.kernel_shape is not None, "kernel_shape must be provided"
        kernel_shape = list(self.kernel_shape)
        assert len(kernel_shape) == spatial_dims, f"kernel_shape length {len(kernel_shape)} must match spatial dims {spatial_dims}"

        strides_spatial = list(self.strides or [1] * spatial_dims)
        assert len(strides_spatial) == spatial_dims, f"strides length {len(strides_spatial)} must match spatial dims {spatial_dims}"

        dilations = [max(d, 1) for d in (self.dilations or [1] * spatial_dims)]
        assert len(dilations) == spatial_dims, f"dilations length {len(dilations)} must match spatial dims {spatial_dims}"

        pads_in = list(self.pads or [0] * (2 * spatial_dims))
        assert len(pads_in) == 2 * spatial_dims, f"pads length {len(pads_in)} must be 2 * spatial dims {2 * spatial_dims}"

        cip = self.count_include_pad or 0
        p = self.p or 2
        inv_p = 1.0 / p

        # 处理 auto_pad / ceil_mode
        auto_pad = self.auto_pad or "NOTSET"
        if auto_pad not in ("NOTSET", ""):
            # ONNX规范：pads 不能与 auto_pad 同时使用
            assert self.pads is None, f"pads cannot be specified when auto_pad is not 'NOTSET'. Got auto_pad='{auto_pad}' and pads={self.pads}"
            # ONNX规范：ceil_mode 不能与 auto_pad 同时使用
            assert self.ceil_mode == 0, f"ceil_mode is not supported with auto_pad. Got auto_pad='{auto_pad}' and ceil_mode={self.ceil_mode}"
            ceil_mode_for_calc = 0
        else:
            ceil_mode_for_calc = int(self.ceil_mode)

        if auto_pad in ("SAME_UPPER", "SAME_LOWER", "VALID"):
            output_spatial_shape = self._get_output_shape_auto_pad(auto_pad, input_spatial_shape, kernel_shape, strides_spatial, dilations)
            pad_shape = self._get_pad_shape(auto_pad, input_spatial_shape, kernel_shape, strides_spatial, output_spatial_shape) if auto_pad != "VALID" else [0] * spatial_dims
            pads_final = self._get_pad_with_auto_pad(auto_pad, pad_shape)
        else:
            output_spatial_shape, pads_final = self._get_output_shape_explicit_padding(input_spatial_shape, kernel_shape, strides_spatial, pads_in, dilations, ceil_mode_for_calc)

        shape_out = [n, c] + list(input_spatial_shape)
        tensor_out = tensor_in

        for dim in range(spatial_dims):
            if kernel_shape[dim] == 1 and strides_spatial[dim] == 1 and pads_final[dim] == 0 and pads_final[
                spatial_dims + dim] == 0:
                # 如果该维度不需要池化，跳过
                continue

            shape_in = shape_out
            tensor_in = tensor_out
            shape_out = [n, c] + [
                output_spatial_shape[i] if i == dim else shape_in[2 + i]
                for i in range(len(shape_in[2:]))
            ]
            tensor_out = self.manager.tensor(np.zeros(int(np.prod(shape_out)), dtype=np.float32))
            updated_tensors.append(tensor_out)

            pre_size = int(np.prod(shape_in[:2 + dim]))  # n * c * spatial[0] * ... * spatial[dim-1]
            post_size = int(np.prod(input_spatial_shape[dim + 1:]))  # spatial[dim+1] * ... * spatial[-1]

            workgroup = (pre_size, output_spatial_shape[dim], post_size)
            spec_consts = [
                post_size,
                input_spatial_shape[dim],  # input_dim
                output_spatial_shape[dim],  # output_dim
                kernel_shape[dim],
                strides_spatial[dim],
                pads_final[dim],
                dilations[dim],
                cip,
                p,
                inv_p
            ]

            updated_algorithms.append(self.manager.algorithm(
                [tensor_in, tensor_out],
                self.pool_shader,
                workgroup,
                spec_consts,
                []
            ))

        return [(tensor_out, shape_out)]
