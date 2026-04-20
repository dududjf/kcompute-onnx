import numpy as np
import kp
from .shader_utils import compile_source


class ResizeOp:
    def __init__(self, manager: kp.Manager,
                 antialias=0,  # 是否抗锯齿，仅在linear和cubic模式下降采样时有效。
                 axes=None,
                 coordinate_transformation_mode='half_pixel',  # 坐标对齐方式，有6种模式: "asymmetric"、"align_corners"、
                 # "half_pixel"、"pytorch_half_pixel"、"tf_half_pixel"和"tf_crop_and_resize"
                 cubic_coeff_a=-0.75,  # 三次插值的系数 a，通常为 -0.75
                 exclude_outside=0,
                 extrapolation_value=0.0,  # 如果坐标超出输入张量范围，用此值填充
                 keep_aspect_ratio_policy='stretch',
                 # 当使用sizes输入时，如何保持宽高比，有三种选择: "stretch"（拉伸）、"not_larger"（不放大）和 "not_smaller"（不缩小）
                 mode='nearest',  # 插值模式，有三种选择: "nearest"（最近邻插值）、"linear"（双线性插值）和"cubic"（双三次插值）
                 nearest_mode='round_prefer_floor'  # 仅当mode为"nearest"时有效，指定如何选取最近像素， 有：
                 # "round_prefer_floor"、"round_prefer_ceil"、"floor" 和 "ceil"
                 ):
        self.manager = manager
        # 初始化默认值（用于 set_attributes 中的比较）
        self.mode = 'nearest'
        self.antialias = 0
        self.coordinate_transformation_mode = 'half_pixel'
        self.nearest_mode = 'round_prefer_floor'
        self.exclude_outside = 0
        self.cubic_coeff_a = -0.75
        self.axes = None
        self.extrapolation_value = 0.0
        self.keep_aspect_ratio_policy = 'stretch'
        
        # 使用 set_attributes 设置传入的参数（参数顺序与 __init__ 保持一致）
        self.set_attributes(
            antialias=antialias,
            axes=axes,
            coordinate_transformation_mode=coordinate_transformation_mode,
            cubic_coeff_a=cubic_coeff_a,
            exclude_outside=exclude_outside,
            extrapolation_value=extrapolation_value,
            keep_aspect_ratio_policy=keep_aspect_ratio_policy,
            mode=mode,
            nearest_mode=nearest_mode
        )

    def _compile_shaders(self):
        """根据当前属性编译 shader"""
        mode = self.mode
        antialias = self.antialias

        # 按需生成 shader（根据 mode 和 antialias）
        if mode == 'nearest':
            self.compiled_shader = compile_source(self._generate_nearest_shader())
            # nearest 模式不支持 antialias，无需创建 antialias shader
        elif mode == 'linear':
            self.compiled_shader = compile_source(self._generate_linear_shader(antialias=False))
            if antialias:
                self.compiled_shader_antialias = compile_source(self._generate_linear_shader(antialias=True))
        elif mode == 'cubic':
            self.compiled_shader = compile_source(self._generate_cubic_shader(antialias=False))
            if antialias:
                self.compiled_shader_antialias = compile_source(self._generate_cubic_shader(antialias=True))

    def set_attributes(self, antialias=None, axes=None, coordinate_transformation_mode=None, cubic_coeff_a=None,
                       exclude_outside=None, extrapolation_value=None, keep_aspect_ratio_policy=None,
                       mode=None, nearest_mode=None):
        """
        修改属性并动态重新编译 shader
        
        支持的属性：
        - antialias: 是否抗锯齿 (0 或 1)
        - axes: 处理的轴
        - coordinate_transformation_mode: 坐标对齐方式
        - cubic_coeff_a: 三次插值系数（仅 cubic 模式有效）
        - exclude_outside: 是否排除外部 (0 或 1)
        - extrapolation_value: 外推值
        - keep_aspect_ratio_policy: 宽高比策略
        - mode: 插值模式 ('nearest', 'linear', 'cubic')
        - nearest_mode: 最近邻模式（仅 nearest 模式有效）
        
        示例:
            op.set_attributes(mode='linear', antialias=1)
            op.set_attributes(coordinate_transformation_mode='align_corners')
        """
        # 检查是否有影响 shader 的属性被修改
        need_recompile = False

        # 逐个检查并更新属性（不使用字典），顺序与 __init__ 保持一致
        if antialias is not None:
            if hasattr(self, 'antialias') and self.antialias != antialias:
                need_recompile = True
            self.antialias = antialias

        if axes is not None:
            self.axes = axes

        if coordinate_transformation_mode is not None:
            if hasattr(self, 'coordinate_transformation_mode') and self.coordinate_transformation_mode != coordinate_transformation_mode:
                need_recompile = True
            self.coordinate_transformation_mode = coordinate_transformation_mode

        if cubic_coeff_a is not None:
            if hasattr(self, 'cubic_coeff_a') and self.cubic_coeff_a != cubic_coeff_a:
                need_recompile = True
            self.cubic_coeff_a = cubic_coeff_a

        if exclude_outside is not None:
            if hasattr(self, 'exclude_outside') and self.exclude_outside != exclude_outside:
                need_recompile = True
            self.exclude_outside = exclude_outside

        if extrapolation_value is not None:
            self.extrapolation_value = extrapolation_value

        if keep_aspect_ratio_policy is not None:
            self.keep_aspect_ratio_policy = keep_aspect_ratio_policy

        if mode is not None:
            if hasattr(self, 'mode') and self.mode != mode:
                need_recompile = True
            self.mode = mode

        if nearest_mode is not None:
            if hasattr(self, 'nearest_mode') and self.nearest_mode != nearest_mode:
                need_recompile = True
            self.nearest_mode = nearest_mode

        # 如果需要，重新编译 shader（第一次调用时总是需要编译）
        if need_recompile or not hasattr(self, 'compiled_shader'):
            self._compile_shaders()

    def _generate_coord_calc(self, coord_var, scale_var, in_size_var, out_size_var, roi_start_var, roi_end_var):
        """根据 coordinate_transformation_mode 生成坐标计算代码（无分支）"""
        mode = self.coordinate_transformation_mode
        if mode == 'half_pixel':
            return f"({coord_var} + 0.5) / {scale_var} - 0.5"
        elif mode == 'asymmetric':
            return f"{coord_var} / {scale_var}"
        elif mode == 'align_corners':
            return f"({out_size_var} == 1.0) ? 0.0 : ({coord_var} * ({in_size_var} - 1.0) / ({out_size_var} - 1.0))"
        elif mode == 'pytorch_half_pixel':
            return f"({out_size_var} == 1.0) ? -0.5 : (({coord_var} + 0.5) / {scale_var} - 0.5)"
        elif mode == 'tf_crop_and_resize':
            return f"(({out_size_var} == 1.0) ? (({roi_end_var} - {roi_start_var}) * ({in_size_var} - 1.0) / 2.0) : " \
                   f"({coord_var} * ({roi_end_var} - {roi_start_var}) * ({in_size_var} - 1.0)" \
                   f" / ({out_size_var} - 1.0))) + {roi_start_var} * ({in_size_var} - 1.0)"
        else:  # half_pixel_symmetric
            return f"({in_size_var} / 2.0) * (1.0 - {out_size_var} / ({scale_var} * {in_size_var})) + " \
                   f"({coord_var} + 0.5) / {scale_var} - 0.5"

    def _generate_nearest_index(self, x_var, in_size_var):
        """根据 nearest_mode 生成最近邻索引计算代码（无分支）"""
        mode = self.nearest_mode
        if mode == 'round_prefer_floor':
            return f"clamp(int(({x_var} == floor({x_var}) + 0.5) ? " \
                   f"floor({x_var}) : round({x_var})), 0, {in_size_var} - 1)"
        elif mode == 'round_prefer_ceil':
            return f"clamp(int(({x_var} == floor({x_var}) + 0.5) ? " \
                   f"ceil({x_var}) : round({x_var})), 0, {in_size_var} - 1)"
        elif mode == 'floor':
            return f"clamp(int(floor({x_var})), 0, {in_size_var} - 1)"
        else:  # ceil
            return f"clamp(int(ceil({x_var})), 0, {in_size_var} - 1)"

    def _generate_extrapolation_check(self):
        """仅在 tf_crop_and_resize 模式下生成 extrapolation 检查代码"""
        if self.coordinate_transformation_mode == 'tf_crop_and_resize':
            return """
    if (in_y_f < 0.0 || in_y_f > float(in_h - 1) || in_x_f < 0.0 || in_x_f > float(in_w - 1)) {
        out_buf[leading_idx * uint(out_h * out_w) + out_y * uint(out_w) + out_x] = extrapolation_f;
        return;
    }"""
        return ""

    def _generate_nearest_shader(self):
        """生成 nearest 模式的 shader"""
        coord_calc_y = self._generate_coord_calc("float(out_y)", "scale_h", "float(in_h)", "float(out_h)",
                                                 "roi_h_start_f", "roi_h_end_f")
        coord_calc_x = self._generate_coord_calc("float(out_x)", "scale_w", "float(in_w)", "float(out_w)",
                                                 "roi_w_start_f", "roi_w_end_f")
        extrapolation_check = self._generate_extrapolation_check()
        nearest_y = self._generate_nearest_index("in_y_f", "in_h")
        nearest_x = self._generate_nearest_index("in_x_f", "in_w")

        return f"""#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  {{ float in_buf[];  }};
layout (binding = 1) writeonly buffer buf_out {{ float out_buf[]; }};

layout (constant_id = 0) const float in_h_f = 1.0;
layout (constant_id = 1) const float in_w_f = 1.0;
layout (constant_id = 2) const float out_h_f = 1.0;
layout (constant_id = 3) const float out_w_f = 1.0;
layout (constant_id = 4) const float scale_h_f = 1.0;
layout (constant_id = 5) const float scale_w_f = 1.0;
layout (constant_id = 6) const float extrapolation_f = 0.0;
layout (constant_id = 7) const float roi_h_start_f = 0.0;
layout (constant_id = 8) const float roi_h_end_f = 1.0;
layout (constant_id = 9) const float roi_w_start_f = 0.0;
layout (constant_id = 10) const float roi_w_end_f = 1.0;

void main() {{
    uint leading_idx = gl_GlobalInvocationID.x;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.z;
    
    int in_h = int(in_h_f);
    int in_w = int(in_w_f);
    int out_h = int(out_h_f);
    int out_w = int(out_w_f);
    float scale_h = scale_h_f;
    float scale_w = scale_w_f;
    
    float in_y_f = {coord_calc_y};
    float in_x_f = {coord_calc_x};
    {extrapolation_check}
    int in_y = {nearest_y};
    int in_x = {nearest_x};
    
    uint out_idx = leading_idx * uint(out_h * out_w) + out_y * uint(out_w) + out_x;
    uint in_idx = leading_idx * uint(in_h * in_w) + uint(in_y * in_w + in_x);
    out_buf[out_idx] = in_buf[in_idx];
}}"""

    def _generate_exclude_outside_code(self):
        """生成 exclude_outside 处理代码"""
        if self.exclude_outside:
            return """
    if (y0 < 0 || y0 >= in_h) cy0 = 0.0;
    if (y1 < 0 || y1 >= in_h) cy1 = 0.0;
    if (x0 < 0 || x0 >= in_w) cx0 = 0.0;
    if (x1 < 0 || x1 >= in_w) cx1 = 0.0;
    
    float sum = (cy0 + cy1) * (cx0 + cx1);
    if (sum > 0.0) {
        float norm = 1.0 / sum;
        cy0 *= norm; cy1 *= norm;
        cx0 *= norm; cx1 *= norm;
    }"""
        return ""

    def _generate_linear_shader(self, antialias=False):
        """生成 linear 模式的 shader"""
        coord_calc_y = self._generate_coord_calc("float(out_y)", "scale_h", "float(in_h)", "float(out_h)",
                                                 "roi_h_start_f", "roi_h_end_f")
        coord_calc_x = self._generate_coord_calc("float(out_x)", "scale_w", "float(in_w)", "float(out_w)",
                                                 "roi_w_start_f", "roi_w_end_f")
        extrapolation_check = self._generate_extrapolation_check()

        if antialias:
            # Antialias 版本使用扩展的三角滤波器
            exclude_check = """
            if (yj < 0 || yj >= in_h || xi < 0 || xi >= in_w) continue;""" if self.exclude_outside else ""

            return f"""#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  {{ float in_buf[];  }};
layout (binding = 1) writeonly buffer buf_out {{ float out_buf[]; }};

layout (constant_id = 0) const float in_h_f = 1.0;
layout (constant_id = 1) const float in_w_f = 1.0;
layout (constant_id = 2) const float out_h_f = 1.0;
layout (constant_id = 3) const float out_w_f = 1.0;
layout (constant_id = 4) const float scale_h_f = 1.0;
layout (constant_id = 5) const float scale_w_f = 1.0;
layout (constant_id = 6) const float extrapolation_f = 0.0;
layout (constant_id = 7) const float roi_h_start_f = 0.0;
layout (constant_id = 8) const float roi_h_end_f = 1.0;
layout (constant_id = 9) const float roi_w_start_f = 0.0;
layout (constant_id = 10) const float roi_w_end_f = 1.0;

float triangle_filter(float x) {{
    float ax = abs(x);
    if (ax < 1.0) return 1.0 - ax;
    return 0.0;
}}

void main() {{
    uint leading_idx = gl_GlobalInvocationID.x;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.z;
    
    int in_h = int(in_h_f);
    int in_w = int(in_w_f);
    int out_h = int(out_h_f);
    int out_w = int(out_w_f);
    float scale_h = scale_h_f;
    float scale_w = scale_w_f;
    
    float in_y_f = {coord_calc_y};
    float in_x_f = {coord_calc_x};
    {extrapolation_check}
    float support_h = max(1.0, 1.0 / scale_h);
    float support_w = max(1.0, 1.0 / scale_w);
    
    int radius_h = int(ceil(support_h));
    int radius_w = int(ceil(support_w));
    
    int center_y = int(round(in_y_f));
    int center_x = int(round(in_x_f));
    
    float result = 0.0;
    float weight_sum = 0.0;
    uint base = leading_idx * uint(in_h * in_w);
    
    for (int dy = -radius_h; dy <= radius_h; ++dy) {{
        int yj = center_y + dy;
        float fy = float(yj) - in_y_f;
        float wy = triangle_filter(fy / support_h) / support_h;
        
        for (int dx = -radius_w; dx <= radius_w; ++dx) {{
            int xi = center_x + dx;
            float fx = float(xi) - in_x_f;
            float wx = triangle_filter(fx / support_w) / support_w;
            
            float w = wx * wy;
            if (w <= 0.0) continue;
            {exclude_check}
            int yc = clamp(yj, 0, in_h - 1);
            int xc = clamp(xi, 0, in_w - 1);
            
            result += w * in_buf[base + uint(yc * in_w + xc)];
            weight_sum += w;
        }}
    }}
    
    if (weight_sum > 0.0) result /= weight_sum;
    
    out_buf[leading_idx * uint(out_h * out_w) + out_y * uint(out_w) + out_x] = result;
}}"""
        else:
            # 普通双线性插值
            exclude_outside_code = self._generate_exclude_outside_code()

            return f"""#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  {{ float in_buf[];  }};
layout (binding = 1) writeonly buffer buf_out {{ float out_buf[]; }};

layout (constant_id = 0) const float in_h_f = 1.0;
layout (constant_id = 1) const float in_w_f = 1.0;
layout (constant_id = 2) const float out_h_f = 1.0;
layout (constant_id = 3) const float out_w_f = 1.0;
layout (constant_id = 4) const float scale_h_f = 1.0;
layout (constant_id = 5) const float scale_w_f = 1.0;
layout (constant_id = 6) const float extrapolation_f = 0.0;
layout (constant_id = 7) const float roi_h_start_f = 0.0;
layout (constant_id = 8) const float roi_h_end_f = 1.0;
layout (constant_id = 9) const float roi_w_start_f = 0.0;
layout (constant_id = 10) const float roi_w_end_f = 1.0;

void main() {{
    uint leading_idx = gl_GlobalInvocationID.x;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.z;
    
    int in_h = int(in_h_f);
    int in_w = int(in_w_f);
    int out_h = int(out_h_f);
    int out_w = int(out_w_f);
    float scale_h = scale_h_f;
    float scale_w = scale_w_f;
    
    float in_y_f = {coord_calc_y};
    float in_x_f = {coord_calc_x};
    {extrapolation_check}
    int y0 = int(floor(in_y_f));
    int x0 = int(floor(in_x_f));
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    
    float ry = in_y_f - float(y0);
    float rx = in_x_f - float(x0);
    
    float cy0 = 1.0 - ry;
    float cy1 = ry;
    float cx0 = 1.0 - rx;
    float cx1 = rx;
    {exclude_outside_code}
    int y0c = clamp(y0, 0, in_h - 1);
    int y1c = clamp(y1, 0, in_h - 1);
    int x0c = clamp(x0, 0, in_w - 1);
    int x1c = clamp(x1, 0, in_w - 1);
    
    uint base = leading_idx * uint(in_h * in_w);
    float v00 = in_buf[base + uint(y0c * in_w + x0c)];
    float v01 = in_buf[base + uint(y0c * in_w + x1c)];
    float v10 = in_buf[base + uint(y1c * in_w + x0c)];
    float v11 = in_buf[base + uint(y1c * in_w + x1c)];
    
    float v = cy0 * (cx0 * v00 + cx1 * v01) + cy1 * (cx0 * v10 + cx1 * v11);
    
    out_buf[leading_idx * uint(out_h * out_w) + out_y * uint(out_w) + out_x] = v;
}}"""

    def _generate_cubic_shader(self, antialias=False):
        """生成 cubic 模式的 shader"""
        coord_calc_y = self._generate_coord_calc("float(out_y)", "scale_h", "float(in_h)", "float(out_h)",
                                                 "roi_h_start_f", "roi_h_end_f")
        coord_calc_x = self._generate_coord_calc("float(out_x)", "scale_w", "float(in_w)", "float(out_w)",
                                                 "roi_w_start_f", "roi_w_end_f")
        extrapolation_check = self._generate_extrapolation_check()
        exclude_check = """
            if (yj < 0 || yj >= in_h || xi < 0 || xi >= in_w) continue;""" if self.exclude_outside else ""
        weight_normalize = "if (weight_sum > 0.0) result /= weight_sum;" if self.exclude_outside else ""

        cubic_kernel = f"""
float cubic_weight(float x) {{
    float a = {self.cubic_coeff_a};
    float abs_x = abs(x);
    if (abs_x <= 1.0) {{
        return (a + 2.0) * abs_x * abs_x * abs_x - (a + 3.0) * abs_x * abs_x + 1.0;
    }} else if (abs_x < 2.0) {{
        return a * abs_x * abs_x * abs_x - 5.0 * a * abs_x * abs_x + 8.0 * a * abs_x - 4.0 * a;
    }}
    return 0.0;
}}"""

        if antialias:
            return f"""#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  {{ float in_buf[];  }};
layout (binding = 1) writeonly buffer buf_out {{ float out_buf[]; }};

layout (constant_id = 0) const float in_h_f = 1.0;
layout (constant_id = 1) const float in_w_f = 1.0;
layout (constant_id = 2) const float out_h_f = 1.0;
layout (constant_id = 3) const float out_w_f = 1.0;
layout (constant_id = 4) const float scale_h_f = 1.0;
layout (constant_id = 5) const float scale_w_f = 1.0;
layout (constant_id = 6) const float extrapolation_f = 0.0;
layout (constant_id = 7) const float roi_h_start_f = 0.0;
layout (constant_id = 8) const float roi_h_end_f = 1.0;
layout (constant_id = 9) const float roi_w_start_f = 0.0;
layout (constant_id = 10) const float roi_w_end_f = 1.0;
{cubic_kernel}

void main() {{
    uint leading_idx = gl_GlobalInvocationID.x;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.z;
    
    int in_h = int(in_h_f);
    int in_w = int(in_w_f);
    int out_h = int(out_h_f);
    int out_w = int(out_w_f);
    float scale_h = scale_h_f;
    float scale_w = scale_w_f;
    
    float in_y_f = {coord_calc_y};
    float in_x_f = {coord_calc_x};
    {extrapolation_check}
    float support_h = max(1.0, 1.0 / scale_h);
    float support_w = max(1.0, 1.0 / scale_w);
    
    int radius_h = int(ceil(2.0 * support_h));
    int radius_w = int(ceil(2.0 * support_w));
    
    int center_y = int(floor(in_y_f));
    int center_x = int(floor(in_x_f));
    float dy = in_y_f - float(center_y);
    float dx = in_x_f - float(center_x);
    
    float result = 0.0;
    float weight_sum = 0.0;
    uint base = leading_idx * uint(in_h * in_w);
    
    for (int j = -radius_h; j <= radius_h; ++j) {{
        int yj = center_y + j;
        float fy = (float(j) - dy) / support_h;
        float wy = cubic_weight(fy) / support_h;
        
        for (int i = -radius_w; i <= radius_w; ++i) {{
            int xi = center_x + i;
            float fx = (float(i) - dx) / support_w;
            float wx = cubic_weight(fx) / support_w;
            
            float w = wx * wy;
            if (w == 0.0) continue;
            {exclude_check}
            int yc = clamp(yj, 0, in_h - 1);
            int xc = clamp(xi, 0, in_w - 1);
            
            result += w * in_buf[base + uint(yc * in_w + xc)];
            weight_sum += w;
        }}
    }}
    
    if (weight_sum > 0.0) result /= weight_sum;
    
    out_buf[leading_idx * uint(out_h * out_w) + out_y * uint(out_w) + out_x] = result;
}}"""
        else:
            return f"""#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_in  {{ float in_buf[];  }};
layout (binding = 1) writeonly buffer buf_out {{ float out_buf[]; }};

layout (constant_id = 0) const float in_h_f = 1.0;
layout (constant_id = 1) const float in_w_f = 1.0;
layout (constant_id = 2) const float out_h_f = 1.0;
layout (constant_id = 3) const float out_w_f = 1.0;
layout (constant_id = 4) const float scale_h_f = 1.0;
layout (constant_id = 5) const float scale_w_f = 1.0;
layout (constant_id = 6) const float extrapolation_f = 0.0;
layout (constant_id = 7) const float roi_h_start_f = 0.0;
layout (constant_id = 8) const float roi_h_end_f = 1.0;
layout (constant_id = 9) const float roi_w_start_f = 0.0;
layout (constant_id = 10) const float roi_w_end_f = 1.0;
{cubic_kernel}

void main() {{
    uint leading_idx = gl_GlobalInvocationID.x;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.z;
    
    int in_h = int(in_h_f);
    int in_w = int(in_w_f);
    int out_h = int(out_h_f);
    int out_w = int(out_w_f);
    float scale_h = scale_h_f;
    float scale_w = scale_w_f;
    
    float in_y_f = {coord_calc_y};
    float in_x_f = {coord_calc_x};
    {extrapolation_check}
    int y_floor = int(floor(in_y_f));
    int x_floor = int(floor(in_x_f));
    float dy = in_y_f - float(y_floor);
    float dx = in_x_f - float(x_floor);
    
    float result = 0.0;
    float weight_sum = 0.0;
    uint base = leading_idx * uint(in_h * in_w);
    
    for (int j = -1; j <= 2; ++j) {{
        int yj = y_floor + j;
        float wy = cubic_weight(dy - float(j));
        
        for (int i = -1; i <= 2; ++i) {{
            int xi = x_floor + i;
            float wx = cubic_weight(dx - float(i));
            float w = wx * wy;
            {exclude_check}
            int yc = clamp(yj, 0, in_h - 1);
            int xc = clamp(xi, 0, in_w - 1);
            
            result += w * in_buf[base + uint(yc * in_w + xc)];
            weight_sum += w;
        }}
    }}
    
    {weight_normalize}
    out_buf[leading_idx * uint(out_h * out_w) + out_y * uint(out_w) + out_x] = result;
}}"""

    def __repr__(self):
        return f"ResizeOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp is None:
                input_tensors.append((None, []))
            else:
                numpy_in = inp.reshape(-1).astype(np.float32)
                tensor = self.manager.tensor(numpy_in)
                input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]] + updated_tensors))
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
        ndim = len(shape_in)

        # 获取 roi, scales, sizes
        roi_data = input_tensors[1][0].data() if len(input_tensors) > 1 and input_tensors[1][0] is not None else None
        scales_data = input_tensors[2][0].data() if len(input_tensors) > 2 and input_tensors[2][0] is not None else None
        sizes_data = input_tensors[3][0].data().astype(int) if len(input_tensors) > 3 and input_tensors[3][
            0] is not None else None

        # 处理 axes 参数
        if self.axes is not None:
            axes = [a if a >= 0 else ndim + a for a in self.axes]
        else:
            axes = list(range(ndim))  # 默认所有维度

        # 计算目标缩放系数
        if sizes_data is not None and len(sizes_data) > 0:
            # sizes 优先
            target_sizes = {axes[i]: int(sizes_data[i]) for i in range(len(sizes_data))}
            target_scales = {ax: float(target_sizes[ax]) / shape_in[ax] for ax in target_sizes}
        elif scales_data is not None and len(scales_data) > 0:
            target_scales = {axes[i]: float(scales_data[i]) for i in range(len(scales_data))}
            target_sizes = {ax: int(round(shape_in[ax] * target_scales[ax])) for ax in target_scales}
        else:
            raise ValueError("Either scales or sizes must be provided")

        # 处理 keep_aspect_ratio_policy
        if self.keep_aspect_ratio_policy != 'stretch' and len(target_scales) >= 2:
            scale_values = list(target_scales.values())
            if self.keep_aspect_ratio_policy == 'not_larger':
                # 使用最小缩放比，确保输出不超过目标尺寸
                uniform_scale = min(scale_values)
            else:  # 'not_smaller'
                # 使用最大缩放比，确保输出不小于目标尺寸
                uniform_scale = max(scale_values)

            # 更新所有轴使用相同缩放比
            for ax in target_scales:
                target_scales[ax] = uniform_scale
                target_sizes[ax] = int(round(shape_in[ax] * uniform_scale))

        # 计算输出形状
        shape_out = list(shape_in)
        scales = [1.0] * ndim
        for ax in target_sizes:
            shape_out[ax] = target_sizes[ax]
            scales[ax] = target_scales[ax]

        # 确定 resize 的两个维度（取 axes 中最后两个，或默认最后两维）
        if len(axes) >= 2:
            h_axis, w_axis = axes[-2], axes[-1]
        else:
            h_axis, w_axis = ndim - 2, ndim - 1

        in_h, in_w = shape_in[h_axis], shape_in[w_axis]
        out_h, out_w = shape_out[h_axis], shape_out[w_axis]
        scale_h, scale_w = scales[h_axis], scales[w_axis]

        # 计算 leading 维度（h_axis 之前的所有维度乘积）
        leading = int(np.prod(shape_in[:h_axis])) if h_axis > 0 else 1

        # 计算 trailing 维度（w_axis 之后的所有维度乘积）
        trailing = int(np.prod(shape_in[w_axis + 1:])) if w_axis < ndim - 1 else 1

        if w_axis != h_axis + 1 or trailing != 1:
            raise ValueError(
                f"ResizeOp currently only supports standard layout where H and W are consecutive last two dimensions. "
                f"Got h_axis={h_axis}, w_axis={w_axis} (w_axis != h_axis + 1: {w_axis != h_axis + 1}), "
                f"trailing={trailing} (trailing != 1: {trailing != 1}). "
                f"Input shape: {shape_in}, axes: {axes if self.axes is not None else 'None'}"
            )

        # 提取 ROI（针对 H, W 维度）
        if roi_data is not None and len(roi_data) == 2 * len(axes):
            h_idx = axes.index(h_axis) if h_axis in axes else len(axes) - 2
            w_idx = axes.index(w_axis) if w_axis in axes else len(axes) - 1
            roi_h_start = roi_data[h_idx]
            roi_w_start = roi_data[w_idx]
            roi_h_end = roi_data[len(axes) + h_idx]
            roi_w_end = roi_data[len(axes) + w_idx]
        else:
            roi_h_start, roi_w_start = 0.0, 0.0
            roi_h_end, roi_w_end = 1.0, 1.0

        out_size = int(np.prod(shape_out))
        tensor_out = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        # 是否使用 antialias（只在缩小时生效）
        use_antialias = self.antialias and (scale_h < 1.0 or scale_w < 1.0)

        # 选择 shader（动态生成的 shader 已在 __init__ 中编译）
        if use_antialias and hasattr(self, 'compiled_shader_antialias'):
            compiled_shader = self.compiled_shader_antialias
        else:
            compiled_shader = self.compiled_shader

        # spec_consts：坐标变换模式、nearest_mode、exclude_outside 等已编译进 shader
        spec_consts = [
            in_h, in_w, out_h, out_w, scale_h, scale_w,
            self.extrapolation_value,
            roi_h_start, roi_h_end, roi_w_start, roi_w_end
        ]

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            compiled_shader,
            (leading, out_h, out_w),
            spec_consts,
            []
        ))

        return [(tensor_out, shape_out)]
