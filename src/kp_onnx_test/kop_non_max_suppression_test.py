import kp
import numpy as np
import time
from kp_onnx_ssbo.kop_non_max_suppression import NonMaxSuppressionOp


# 复制 op_non_max_suppression.py 中的参考实现
def max_min(lhs, rhs):
    if lhs >= rhs:
        return rhs, lhs
    return lhs, rhs


def suppress_by_iou(boxes_data, box_index1, box_index2, center_point_box, iou_threshold):
    box1 = boxes_data[box_index1]
    box2 = boxes_data[box_index2]

    if center_point_box == 0:
        x1_min, x1_max = max_min(box1[1], box1[3])
        x2_min, x2_max = max_min(box2[1], box2[3])

        intersection_x_min = max(x1_min, x2_min)
        intersection_x_max = min(x1_max, x2_max)
        if intersection_x_max <= intersection_x_min:
            return False

        y1_min, y1_max = max_min(box1[0], box1[2])
        y2_min, y2_max = max_min(box2[0], box2[2])
        intersection_y_min = max(y1_min, y2_min)
        intersection_y_max = min(y1_max, y2_max)
        if intersection_y_max <= intersection_y_min:
            return False
    else:
        box1_width_half = box1[2] / 2
        box1_height_half = box1[3] / 2
        box2_width_half = box2[2] / 2
        box2_height_half = box2[3] / 2

        x1_min = box1[0] - box1_width_half
        x1_max = box1[0] + box1_width_half
        x2_min = box2[0] - box2_width_half
        x2_max = box2[0] + box2_width_half

        intersection_x_min = max(x1_min, x2_min)
        intersection_x_max = min(x1_max, x2_max)
        if intersection_x_max <= intersection_x_min:
            return False

        y1_min = box1[1] - box1_height_half
        y1_max = box1[1] + box1_height_half
        y2_min = box2[1] - box2_height_half
        y2_max = box2[1] + box2_height_half

        intersection_y_min = max(y1_min, y2_min)
        intersection_y_max = min(y1_max, y2_max)
        if intersection_y_max <= intersection_y_min:
            return False

    intersection_area = (intersection_x_max - intersection_x_min) * (
        intersection_y_max - intersection_y_min
    )
    if intersection_area <= 0:
        return False

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersection_area

    if area1 <= 0 or area2 <= 0 or union_area <= 0:
        return False

    intersection_over_union = intersection_area / union_area
    return intersection_over_union > iou_threshold


class BoxInfo:
    def __init__(self, score=0, idx=-1):
        self.score_ = score
        self.idx_ = idx

    def __lt__(self, rhs):
        return self.score_ < rhs.score_ or (
            self.score_ == rhs.score_ and self.idx_ > rhs.idx_
        )


def onnx_nms(boxes, scores, max_output_boxes_per_class=None, iou_threshold=None,
             score_threshold=None, center_point_box=0):
    """ONNX NMS 参考实现"""

    # 默认值
    if max_output_boxes_per_class is None or max_output_boxes_per_class.size == 0:
        max_output_boxes_per_class = 0
    else:
        max_output_boxes_per_class = int(max(max_output_boxes_per_class[0], 0))

    if iou_threshold is None or iou_threshold.size == 0:
        iou_threshold = 0.0
    else:
        iou_threshold = float(iou_threshold[0])

    if score_threshold is None or score_threshold.size == 0:
        score_threshold = 0.0
    else:
        score_threshold = float(score_threshold[0])

    num_batches = boxes.shape[0]
    num_boxes = boxes.shape[1]
    num_classes = scores.shape[1]

    if max_output_boxes_per_class == 0:
        max_output_boxes_per_class = num_boxes

    selected_indices = []

    for batch_index in range(num_batches):
        for class_index in range(num_classes):
            batch_boxes = boxes[batch_index]
            class_scores = scores[batch_index, class_index]

            # 过滤低分 boxes
            candidate_boxes = []
            for box_index in range(num_boxes):
                if class_scores[box_index] > score_threshold:
                    candidate_boxes.append(BoxInfo(class_scores[box_index], box_index))

            sorted_boxes = sorted(candidate_boxes)

            selected_boxes_inside_class = []
            while (len(sorted_boxes) > 0 and
                   len(selected_boxes_inside_class) < max_output_boxes_per_class):
                next_top_score = sorted_boxes[-1]

                selected = True
                for selected_index in selected_boxes_inside_class:
                    if suppress_by_iou(
                        batch_boxes,
                        next_top_score.idx_,
                        selected_index.idx_,
                        center_point_box,
                        iou_threshold
                    ):
                        selected = False
                        break

                if selected:
                    selected_boxes_inside_class.append(next_top_score)
                    selected_indices.append([batch_index, class_index, next_top_score.idx_])

                sorted_boxes.pop()

    if not selected_indices:
        return np.empty((0, 3), dtype=np.int64)

    return np.array(selected_indices, dtype=np.int64)


# 测试
mgr = kp.Manager()
print(mgr.get_device_properties())

nms_op = NonMaxSuppressionOp(mgr, center_point_box=0)

# Case 1: 基本测试，center_point_box=0，有 score_threshold
print("Case 1: Basic test, center_point_box=0, with score_threshold")
np.random.seed(42)
boxes = np.array([[[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9], [0, 10, 1, 11]]], dtype=np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95]]], dtype=np.float32)
max_output_boxes_per_class = np.array([3], dtype=np.int64)
iou_threshold = np.array([0.5], dtype=np.float32)
score_threshold = np.array([0.0], dtype=np.float32)

t0 = time.time()
np_out = onnx_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")
print("NumPy result:\n", np_out)

t0 = time.time()
kp_out = nms_op.run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)[0]
print(f"{nms_op}:", kp_out.shape, time.time() - t0, "seconds")
print("Kompute result:\n", kp_out)

print("shape equal:", kp_out.shape == np_out.shape)
if kp_out.shape == np_out.shape and np_out.size > 0:
    print("Results equal:", np.array_equal(np_out, kp_out))
print("----")

# Case 2: center_point_box=1
print("Case 2: center_point_box=1")
np.random.seed(42)
boxes = np.array([[[0.5, 0.5, 1.0, 1.0], [0.5, 0.6, 1.0, 1.0], [0.5, 0.4, 1.0, 1.0], [5.5, 5.5, 1.0, 1.0]]], dtype=np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95]]], dtype=np.float32)
max_output_boxes_per_class = np.array([3], dtype=np.int64)
iou_threshold = np.array([0.5], dtype=np.float32)
score_threshold = np.array([0.0], dtype=np.float32)

nms_op_center1 = NonMaxSuppressionOp(mgr, center_point_box=1)

t0 = time.time()
np_out = onnx_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box=1)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")
print("NumPy result:\n", np_out)

t0 = time.time()
kp_out = nms_op_center1.run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)[0]
print(f"{nms_op_center1}:", kp_out.shape, time.time() - t0, "seconds")
print("Kompute result:\n", kp_out)

print("shape equal:", kp_out.shape == np_out.shape)
if kp_out.shape == np_out.shape and np_out.size > 0:
    print("Results equal:", np.array_equal(np_out, kp_out))
print("----")

# Case 3: 多 batch 多 class
print("Case 3: Multiple batches and classes")
np.random.seed(42)
boxes = np.random.uniform(0, 10, (2, 10, 4)).astype(np.float32)
scores = np.random.uniform(0, 1, (2, 3, 10)).astype(np.float32)
max_output_boxes_per_class = np.array([5], dtype=np.int64)
iou_threshold = np.array([0.5], dtype=np.float32)
score_threshold = np.array([0.4], dtype=np.float32)

t0 = time.time()
np_out = onnx_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
kp_out = nms_op.run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)[0]
print(f"{nms_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
if kp_out.shape == np_out.shape and np_out.size > 0:
    print("Results equal:", np.array_equal(np_out, kp_out))
print("----")

# Case 4: 空结果测试（所有分数低于阈值）
print("Case 4: Empty result (all scores below threshold)")
boxes = np.random.uniform(0, 10, (1, 5, 4)).astype(np.float32)
scores = np.array([[[0.1, 0.2, 0.15, 0.1, 0.05]]], dtype=np.float32)
max_output_boxes_per_class = np.array([3], dtype=np.int64)
iou_threshold = np.array([0.5], dtype=np.float32)
score_threshold = np.array([0.5], dtype=np.float32)

t0 = time.time()
np_out = onnx_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")

t0 = time.time()
kp_out = nms_op.run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)[0]
print(f"{nms_op}:", kp_out.shape, time.time() - t0, "seconds")

print("shape equal:", kp_out.shape == np_out.shape)
print("----")

# Case 5: max_output_boxes_per_class=0（不限制，触发 316-317 行 == 0 → num_boxes 转换）
print("Case 5: max_output_boxes_per_class=0 (no limit, triggers line 316-317)")
np.random.seed(7)
boxes_5 = np.random.uniform(0, 10, (1, 6, 4)).astype(np.float32)
scores_5 = np.random.uniform(0, 1, (1, 2, 6)).astype(np.float32)
max_output_boxes_5 = np.array([0], dtype=np.int64)   # 0 表示不限制
iou_threshold_5    = np.array([0.5], dtype=np.float32)
score_threshold_5  = np.array([0.3], dtype=np.float32)

t0 = time.time()
np_out = onnx_nms(boxes_5, scores_5, max_output_boxes_5, iou_threshold_5, score_threshold_5, center_point_box=0)
print("NumPy:", np_out.shape, time.time() - t0, "seconds")
print("NumPy result:\n", np_out)

t0 = time.time()
kp_out = nms_op.run(boxes_5, scores_5, max_output_boxes_5, iou_threshold_5, score_threshold_5)[0]
print(f"{nms_op}:", kp_out.shape, time.time() - t0, "seconds")
print("Kompute result:\n", kp_out)

print("shape equal:", kp_out.shape == np_out.shape)
if kp_out.shape == np_out.shape and np_out.size > 0:
    print("Results equal:", np.array_equal(np_out, kp_out))
print("----")

