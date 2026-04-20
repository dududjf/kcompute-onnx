import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_2D, LOCAL_Y_2D, LOCAL_X_3D, LOCAL_Y_3D, LOCAL_Z_3D


class NonMaxSuppressionOp:

    def __init__(self, manager: kp.Manager, center_point_box=0):
        self.center_point_box = center_point_box
        self.manager = manager

        # NMS Shader
        # Bindings:
        #   0: InScores     (readonly)
        #   1: InIoU        (readonly)
        #   2: OutCompact   (writeonly)
        #   3: Params       (readonly) — last binding
        # Params layout (uint32):
        #   [0] num_boxes
        #   [1] max_output
        #   [2] score_threshold (bits of float32)
        #   [3] iou_threshold   (bits of float32)
        #   [4] num_classes
        #   [5] num_batches
        self.nms_shader = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InScores   {{ float scores[]; }};
layout(std430, set = 0, binding = 1) readonly  buffer InIoU      {{ float iou_matrix[]; }};
layout(std430, set = 0, binding = 2) writeonly buffer OutCompact {{ float compact_output[]; }};
layout(std430, set = 0, binding = 3) readonly  buffer Params     {{ uint params[]; }};

void main() {{
    uint class_idx = gl_GlobalInvocationID.x;
    uint batch_idx = gl_GlobalInvocationID.y;

    uint num_boxes      = params[0];
    uint max_output     = params[1];
    float score_threshold = uintBitsToFloat(params[2]);
    float iou_threshold   = uintBitsToFloat(params[3]);
    uint num_classes    = params[4];
    uint num_batches    = params[5];

    if (class_idx >= num_classes || batch_idx >= num_batches) return;

    uint num_classes_x_num_boxes    = num_classes * num_boxes;
    uint num_boxes_sq               = num_boxes * num_boxes;
    uint max_output_x_3             = max_output * 3u;
    uint num_classes_x_max_output_x_3 = num_classes * max_output_x_3;

    uint global_score_base = batch_idx * num_classes_x_num_boxes + class_idx * num_boxes;
    uint iou_batch_offset  = batch_idx * num_boxes_sq;
    uint out_base          = batch_idx * num_classes_x_max_output_x_3 + class_idx * max_output_x_3;

    // 初始化输出为 -1（标记无效）
    for (uint i = 0u; i < max_output; i++) {{
        uint wi = out_base + i * 3u;
        compact_output[wi]      = -1.0;
        compact_output[wi + 1u] = -1.0;
        compact_output[wi + 2u] = -1.0;
    }}

    // Greedy NMS — 用寄存器数组记录已选 box（上限 256）
    uint selected_count = 0u;
    uint selected_boxes[256];
    for (uint i = 0u; i < 256u; i++) {{
        selected_boxes[i] = 0xFFFFFFFFu;
    }}

    for (uint iter = 0u; iter < max_output && iter < num_boxes; iter++) {{
        float best_score = score_threshold;
        uint  best_box   = 0xFFFFFFFFu;

        for (uint box_id = 0u; box_id < num_boxes; box_id++) {{
            float score = scores[global_score_base + box_id];
            if (score <= best_score) continue;

            // 是否已选
            bool already = false;
            for (uint s = 0u; s < selected_count; s++) {{
                if (selected_boxes[s] == box_id) {{ already = true; break; }}
            }}
            if (already) continue;

            // 是否被已选 box 抑制
            bool suppressed = false;
            uint iou_row = iou_batch_offset + box_id * num_boxes;
            for (uint s = 0u; s < selected_count; s++) {{
                if (iou_matrix[iou_row + selected_boxes[s]] > iou_threshold) {{
                    suppressed = true;
                    break;
                }}
            }}
            if (suppressed) continue;

            best_score = score;
            best_box   = box_id;
        }}

        if (best_box == 0xFFFFFFFFu) break;

        selected_boxes[selected_count] = best_box;
        uint wi = out_base + selected_count * 3u;
        compact_output[wi]      = float(batch_idx);
        compact_output[wi + 1u] = float(class_idx);
        compact_output[wi + 2u] = float(best_box);
        selected_count++;
    }}
}}
""")

        # IoU Shader — Corner Point Box (center_point_box == 0)
        # ONNX corner box 格式: [y1, x1, y2, x2]
        # Bindings:
        #   0: InBoxes  (readonly)
        #   1: OutIoU   (writeonly)
        #   2: Params   (readonly) — last binding
        # Params layout (uint32):
        #   [0] num_boxes
        #   [1] num_batches
        self.iou_shader_corner = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBoxes {{ float boxes[]; }};
layout(std430, set = 0, binding = 1) writeonly buffer OutIoU  {{ float iou_matrix[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer Params  {{ uint params[]; }};

void main() {{
    uint box_i     = gl_GlobalInvocationID.x;
    uint box_j     = gl_GlobalInvocationID.y;
    uint batch_idx = gl_GlobalInvocationID.z;

    uint num_boxes  = params[0];
    uint num_batches = params[1];

    if (box_i >= num_boxes || box_j >= num_boxes || batch_idx >= num_batches) return;

    uint iou_idx = batch_idx * num_boxes * num_boxes + box_i * num_boxes + box_j;

    if (box_i == box_j) {{
        iou_matrix[iou_idx] = 0.0;
        return;
    }}

    uint base = batch_idx * num_boxes * 4u;
    uint idx1 = base + box_i * 4u;
    uint idx2 = base + box_j * 4u;

    // ONNX corner format: [y1, x1, y2, x2]
    float y1a = boxes[idx1],     x1a = boxes[idx1 + 1u];
    float y2a = boxes[idx1 + 2u], x2a = boxes[idx1 + 3u];
    float x1_min = min(x1a, x2a); float x1_max = max(x1a, x2a);
    float y1_min = min(y1a, y2a); float y1_max = max(y1a, y2a);

    float y1b = boxes[idx2],     x1b = boxes[idx2 + 1u];
    float y2b = boxes[idx2 + 2u], x2b = boxes[idx2 + 3u];
    float x2_min = min(x1b, x2b); float x2_max = max(x1b, x2b);
    float y2_min = min(y1b, y2b); float y2_max = max(y1b, y2b);

    float inter_x_min = max(x1_min, x2_min);
    float inter_x_max = min(x1_max, x2_max);
    float inter_y_min = max(y1_min, y2_min);
    float inter_y_max = min(y1_max, y2_max);

    float iw = inter_x_max - inter_x_min;
    float ih = inter_y_max - inter_y_min;

    float iou = 0.0;
    if (iw > 0.0 && ih > 0.0) {{
        float inter_area = iw * ih;
        float area1 = (x1_max - x1_min) * (y1_max - y1_min);
        float area2 = (x2_max - x2_min) * (y2_max - y2_min);
        float union_area = area1 + area2 - inter_area;
        if (union_area > 0.0) iou = inter_area / union_area;
    }}

    iou_matrix[iou_idx] = iou;
}}
""")

        # IoU Shader — Center Point Box (center_point_box != 0)
        # ONNX center box 格式: [cx, cy, w, h]
        self.iou_shader_center = compile_source(f"""
#version 450
layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in;

layout(std430, set = 0, binding = 0) readonly  buffer InBoxes {{ float boxes[]; }};
layout(std430, set = 0, binding = 1) writeonly buffer OutIoU  {{ float iou_matrix[]; }};
layout(std430, set = 0, binding = 2) readonly  buffer Params  {{ uint params[]; }};

void main() {{
    uint box_i     = gl_GlobalInvocationID.x;
    uint box_j     = gl_GlobalInvocationID.y;
    uint batch_idx = gl_GlobalInvocationID.z;

    uint num_boxes   = params[0];
    uint num_batches = params[1];

    if (box_i >= num_boxes || box_j >= num_boxes || batch_idx >= num_batches) return;

    uint iou_idx = batch_idx * num_boxes * num_boxes + box_i * num_boxes + box_j;

    if (box_i == box_j) {{
        iou_matrix[iou_idx] = 0.0;
        return;
    }}

    uint base = batch_idx * num_boxes * 4u;
    uint idx1 = base + box_i * 4u;
    uint idx2 = base + box_j * 4u;

    // ONNX center format: [cx, cy, w, h]
    float cx1 = boxes[idx1],      cy1 = boxes[idx1 + 1u];
    float w1  = boxes[idx1 + 2u], h1  = boxes[idx1 + 3u];
    float x1_min = cx1 - w1 * 0.5; float x1_max = cx1 + w1 * 0.5;
    float y1_min = cy1 - h1 * 0.5; float y1_max = cy1 + h1 * 0.5;

    float cx2 = boxes[idx2],      cy2 = boxes[idx2 + 1u];
    float w2  = boxes[idx2 + 2u], h2  = boxes[idx2 + 3u];
    float x2_min = cx2 - w2 * 0.5; float x2_max = cx2 + w2 * 0.5;
    float y2_min = cy2 - h2 * 0.5; float y2_max = cy2 + h2 * 0.5;

    float inter_x_min = max(x1_min, x2_min);
    float inter_x_max = min(x1_max, x2_max);
    float inter_y_min = max(y1_min, y2_min);
    float inter_y_max = min(y1_max, y2_max);

    float iw = inter_x_max - inter_x_min;
    float ih = inter_y_max - inter_y_min;

    float iou = 0.0;
    if (iw > 0.0 && ih > 0.0) {{
        float inter_area = iw * ih;
        float area1 = w1 * h1;
        float area2 = w2 * h2;
        float union_area = area1 + area2 - inter_area;
        if (union_area > 0.0) iou = inter_area / union_area;
    }}

    iou_matrix[iou_idx] = iou;
}}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"NonMaxSuppressionOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        boxes  = inputs[0]
        scores = inputs[1]

        max_output_boxes_per_class = 0
        iou_threshold   = 0.0
        score_threshold = 0.0

        if len(inputs) > 2:
            v = inputs[2]
            max_output_boxes_per_class = int(v.flatten()[0]) if isinstance(v, np.ndarray) else int(v)
        if len(inputs) > 3:
            v = inputs[3]
            iou_threshold = float(v.flatten()[0]) if isinstance(v, np.ndarray) else float(v)
        if len(inputs) > 4:
            v = inputs[4]
            score_threshold = float(v.flatten()[0]) if isinstance(v, np.ndarray) else float(v)

        boxes_tensor  = self.manager.tensor(boxes.reshape(-1).astype(np.float32))
        scores_tensor = self.manager.tensor(scores.reshape(-1).astype(np.float32))

        input_tensors = [
            (boxes_tensor,  list(boxes.shape)),
            (scores_tensor, list(scores.shape)),
            (self.manager.tensor(np.array([max_output_boxes_per_class], dtype=np.int64)), [1]),
            (self.manager.tensor(np.array([iou_threshold],   dtype=np.float32)), [1]),
            (self.manager.tensor(np.array([score_threshold], dtype=np.float32)), [1]),
        ]

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t for t, _ in input_tensors] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        raw_output = tensor_out.data().reshape(output_shape)
        output = raw_output[raw_output[:, 0] >= 0].astype(np.int64)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors

        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        boxes_tensor,  boxes_shape  = input_tensors[0]
        scores_tensor, scores_shape = input_tensors[1]

        max_output_boxes_per_class = int(input_tensors[2][0].data()[0])
        iou_threshold   = float(input_tensors[3][0].data()[0])
        score_threshold = float(input_tensors[4][0].data()[0])

        num_batches = boxes_shape[0]
        num_boxes   = boxes_shape[1]
        num_classes = scores_shape[1]

        # max_output_boxes_per_class == 0 表示不限制，取 num_boxes
        if max_output_boxes_per_class == 0:
            max_output_boxes_per_class = num_boxes
        # 上限 256（与 NMS shader 寄存器数组大小一致）
        max_output_boxes_per_class = min(max_output_boxes_per_class, 256)

        # ── Step 1: 计算 IoU 矩阵 ─────────────────────────────────────
        iou_size   = num_batches * num_boxes * num_boxes
        iou_tensor = self.manager.tensor(np.zeros(iou_size, dtype=np.float32))
        updated_tensors.append(iou_tensor)

        params_iou = self.manager.tensor_t(
            np.array([num_boxes, num_batches], dtype=np.uint32),
            kp.TensorTypes.device
        )
        self.manager.sequence().record(kp.OpTensorSyncDevice([params_iou])).eval()

        iou_shader = self.iou_shader_corner if self.center_point_box == 0 else self.iou_shader_center
        gx = (num_boxes   + LOCAL_X_3D - 1) // LOCAL_X_3D
        gy = (num_boxes   + LOCAL_Y_3D - 1) // LOCAL_Y_3D
        gz = (num_batches + LOCAL_Z_3D - 1) // LOCAL_Z_3D
        updated_algorithms.append(self.manager.algorithm(
            [boxes_tensor, iou_tensor, params_iou],
            iou_shader,
            (gx, gy, gz)
        ))

        # ── Step 2: NMS 选择 ─────────────────────────────────────────
        out_size       = num_batches * num_classes * max_output_boxes_per_class * 3
        compact_tensor = self.manager.tensor(np.zeros(out_size, dtype=np.float32))
        updated_tensors.append(compact_tensor)

        params_nms = self.manager.tensor_t(
            np.array([
                num_boxes,
                max_output_boxes_per_class,
                np.float32(score_threshold).view(np.uint32),
                np.float32(iou_threshold).view(np.uint32),
                num_classes,
                num_batches,
            ], dtype=np.uint32),
            kp.TensorTypes.device
        )
        self.manager.sequence().record(kp.OpTensorSyncDevice([params_nms])).eval()

        gnx = (num_classes + LOCAL_X_2D - 1) // LOCAL_X_2D
        gny = (num_batches + LOCAL_Y_2D - 1) // LOCAL_Y_2D
        updated_algorithms.append(self.manager.algorithm(
            [scores_tensor, iou_tensor, compact_tensor, params_nms],
            self.nms_shader,
            (gnx, gny, 1)
        ))

        return [(compact_tensor, [out_size // 3, 3])]

