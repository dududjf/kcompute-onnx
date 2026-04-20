import numpy as np
import kp
from .shader_utils import compile_source, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class Col2imOp:
    def __init__(self, manager: kp.Manager, dilations=None, pads=None, strides=None):
        self.manager = manager
        self.dilations = dilations
        self.pads = pads
        self.strides = strides
        self.compiled_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout (std430, set=0, binding = 0) readonly buffer InBuf {{ float in_tensor[]; }};
layout (std430, set=0, binding = 1) buffer OutBuf {{ float out_tensor[]; }};
layout (std430, set=0, binding = 2) readonly buffer UIParams {{ uint params[]; }};

void main() {{
    uint C = params[0], H = params[1], W = params[2];
    uint kernel_h = params[3], kernel_w = params[4];
    uint stride_h = params[5], stride_w = params[6];
    uint pad_h = params[7], pad_w = params[8];
    uint dilation_h = params[9], dilation_w = params[10];
    uint height_col = params[11], width_col = params[12];
    uint n = params[13];

    uint c = gl_GlobalInvocationID.x;
    uint h_out = gl_GlobalInvocationID.y;
    uint w_out = gl_GlobalInvocationID.z;
    if (c >= C || h_out >= H || w_out >= W) return;

    float sum = 0.0;
    uint kernel_size = kernel_h * kernel_w;
    uint L = height_col * width_col;

    for (uint kh = 0; kh < kernel_h; ++kh) {{
        for (uint kw = 0; kw < kernel_w; ++kw) {{
            uint kh_offset = kh * dilation_h;
            uint kw_offset = kw * dilation_w;

            if (h_out + pad_h >= kh_offset && w_out + pad_w >= kw_offset) {{
                uint h_padded = h_out + pad_h - kh_offset;
                uint w_padded = w_out + pad_w - kw_offset;

                if (h_padded % stride_h == 0 && w_padded % stride_w == 0) {{
                    uint h_col = h_padded / stride_h;
                    uint w_col = w_padded / stride_w;

                    if (h_col < height_col && w_col < width_col) {{
                        uint c_col = kh * kernel_w + kw;
                        uint col = h_col * width_col + w_col;
                        uint data_idx = n * (C * kernel_size * L) + (c * kernel_size + c_col) * L + col;
                        sum += in_tensor[data_idx];
                    }}
                }}
            }}
        }}
    }}

    uint out_idx = n * (C * H * W) + c * (H * W) + h_out * W + w_out;
    out_tensor[out_idx] = sum;
}}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"Col2imOp({device_name})"

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
        tensor_data, shape_data = input_tensors[0]
        image_shape = input_tensors[1][0].data().astype(int).tolist()
        block_shape = input_tensors[2][0].data().astype(int).tolist()

        assert len(image_shape) == 2, f"Only 2D image_shape supported, got {len(image_shape)}D"
        assert len(block_shape) == 2, f"Only 2D block_shape supported, got {len(block_shape)}D"

        n_dims = len(image_shape)

        dilations = self.dilations if self.dilations is not None else [1] * n_dims
        pads = self.pads if self.pads is not None else [0] * (2 * n_dims)
        strides = self.strides if self.strides is not None else [1] * n_dims

        N = shape_data[0]
        block_size = int(np.prod(block_shape))
        C = shape_data[1] // block_size
        L = shape_data[2]

        H, W = image_shape
        kernel_h, kernel_w = block_shape
        stride_h, stride_w = strides
        dilation_h, dilation_w = dilations
        pad_h_begin, pad_w_begin = pads[0], pads[1]
        pad_h_end, pad_w_end = pads[n_dims], pads[n_dims + 1]

        height_col = (H + pad_h_begin + pad_h_end - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        width_col = (W + pad_w_begin + pad_w_end - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        expected_L = height_col * width_col
        assert L == expected_L, f"Input L={L} doesn't match expected blocks {height_col}*{width_col}={expected_L}"

        output_shape = [N, C, H, W]
        output_size = int(np.prod(output_shape))
        tensor_out = self.manager.tensor(np.zeros(output_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        group_x = (C + LOCAL_X_3D - 1) // LOCAL_X_3D
        group_y = (H + LOCAL_Y_3D - 1) // LOCAL_Y_3D
        group_z = (W + LOCAL_Z_3D - 1) // LOCAL_Z_3D
        workgroup = (group_x, group_y, group_z)

        for n in range(N):
            params = np.array([
                C, H, W,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h_begin, pad_w_begin,
                dilation_h, dilation_w,
                height_col, width_col,
                n
            ], dtype=np.uint32)

            param_in = self.manager.tensor_t(params, kp.TensorTypes.device)
            self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()

            updated_algorithms.append(self.manager.algorithm(
                [tensor_data, tensor_out, param_in],
                self.compiled_shader,
                workgroup
            ))

        return [(tensor_out, output_shape)]
