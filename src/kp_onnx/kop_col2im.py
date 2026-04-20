import numpy as np
import kp
from .shader_utils import compile_source


class Col2imOp:
    def __init__(self, manager: kp.Manager, dilations=None, pads=None, strides=None):
        self.manager = manager
        self.dilations = dilations
        self.pads = pads
        self.strides = strides
        self.compiled_shader = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_in { float data[]; };
layout (binding = 1) writeonly buffer buf_out { float output_data[]; };

layout (constant_id = 0) const float C_f = 0;
layout (constant_id = 1) const float H_f = 0;
layout (constant_id = 2) const float W_f = 0;
layout (constant_id = 3) const float kernel_h_f = 0;
layout (constant_id = 4) const float kernel_w_f = 0;
layout (constant_id = 5) const float stride_h_f = 0;
layout (constant_id = 6) const float stride_w_f = 0;
layout (constant_id = 7) const float pad_h_f = 0;
layout (constant_id = 8) const float pad_w_f = 0;
layout (constant_id = 9) const float dilation_h_f = 0;
layout (constant_id = 10) const float dilation_w_f = 0;
layout (constant_id = 11) const float height_col_f = 0;
layout (constant_id = 12) const float width_col_f = 0;
layout (constant_id = 13) const float n_f = 0;  // 当前batch索引

void main() {
    uint c = gl_GlobalInvocationID.x;      // channel
    uint h_out = gl_GlobalInvocationID.y;  // 输出像素行索引
    uint w_out = gl_GlobalInvocationID.z;  // 输出像素列索引
    
    uint C = uint(C_f);
    uint H = uint(H_f);
    uint W = uint(W_f);
    uint kernel_h = uint(kernel_h_f);
    uint kernel_w = uint(kernel_w_f);
    uint stride_h = uint(stride_h_f);
    uint stride_w = uint(stride_w_f);
    uint pad_h = uint(pad_h_f);
    uint pad_w = uint(pad_w_f);
    uint dilation_h = uint(dilation_h_f);
    uint dilation_w = uint(dilation_w_f);
    uint height_col = uint(height_col_f);
    uint width_col = uint(width_col_f);
    uint n = uint(n_f);                    // batch索引
    
    float sum = 0.0;
    
    uint kernel_size = kernel_h * kernel_w;
    uint L = height_col * width_col;  //输出像素总数
    
    for (uint kh = 0; kh < kernel_h; ++kh) {
        for (uint kw = 0; kw < kernel_w; ++kw) {
            // 先判断比较 h_out + pad_h 和 kh * dilation_h 的大小
            uint kh_offset = kh * dilation_h;
            uint kw_offset = kw * dilation_w;
            
            // 只有当 h_out + pad_h >= kh_offset 时才继续计算
            if (h_out + pad_h >= kh_offset && w_out + pad_w >= kw_offset) {
                // 反向计算：哪个滑动块在这个 kernel 位置覆盖了当前像素？
                uint h_padded = h_out + pad_h - kh_offset;
                uint w_padded = w_out + pad_w - kw_offset;
                
                // 检查是否能整除 stride (否则不存在对应的输入位置)
                if (h_padded % stride_h == 0 && w_padded % stride_w == 0) {
                    uint h_col = h_padded / stride_h;
                    uint w_col = w_padded / stride_w;
                    
                    // 检查是否在有效的滑动块范围内
                    if (h_col < height_col && w_col < width_col) {
                        // 计算输入索引并累加值
                        uint c_col = kh * kernel_w + kw;
                        uint col = h_col * width_col + w_col;
                        uint data_idx = n * (C * kernel_size * L) + (c * kernel_size + c_col) * L + col;
                        sum += data[data_idx];
                    }
                }
            }
        }
    }
    
    // output layout: [N, C, H, W]
    uint out_idx = n * (C * H * W) + c * (H * W) + h_out * W + w_out;
    output_data[out_idx] = sum;
}
""")

    def __repr__(self):
        return f"Col2imOp({self.manager.get_device_properties()['device_name']})"

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
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
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
        
        # Only support 2D for now
        assert len(image_shape) == 2, f"Only 2D image_shape supported, got {len(image_shape)}D"
        assert len(block_shape) == 2, f"Only 2D block_shape supported, got {len(block_shape)}D"
        
        n_dims = len(image_shape)
        
        # Default parameters
        dilations = self.dilations if self.dilations is not None else [1] * n_dims
        pads = self.pads if self.pads is not None else [0] * (2 * n_dims)
        strides = self.strides if self.strides is not None else [1] * n_dims
        
        # Parse input shape: [N, C * block_size, L]
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
        
        # Calculate column dimensions
        height_col = (H + pad_h_begin + pad_h_end - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        width_col = (W + pad_w_begin + pad_w_end - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        
        # Verify L matches expected number of blocks
        expected_L = height_col * width_col
        assert L == expected_L, f"Input L={L} doesn't match expected blocks {height_col}*{width_col}={expected_L}"
        
        # Output shape: [N, C, H, W]
        output_shape = [N, C, H, W]
        output_size = int(np.prod(output_shape))
        tensor_out = self.manager.tensor(np.zeros(output_size, dtype=np.float32))
        updated_tensors.append(tensor_out)
        
        # 为每个batch创建独立的algorithm
        # workgroup: (C, H, W) - h和w各占一个轴，避免除法取余
        for n in range(N):
            spec_consts = [
                C, H, W,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h_begin, pad_w_begin,
                dilation_h, dilation_w,
                height_col, width_col,
                n  # 当前batch索引
            ]
            
            updated_algorithms.append(self.manager.algorithm(
                [tensor_data, tensor_out],
                self.compiled_shader,
                (C, H, W),  # h和w各占一个轴
                spec_consts,
                []
            ))
        
        return [(tensor_out, output_shape)]

