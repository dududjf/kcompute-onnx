import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D

class TreeEnsembleClassifierOp:

    def __init__(self,
                 manager: kp.Manager,
                 base_values=None,
                 class_ids=None,
                 class_nodeids=None,
                 class_treeids=None,
                 class_weights=None,
                 classlabels_int64s=None,
                 classlabels_strings=None,
                 nodes_falsenodeids=None,
                 nodes_featureids=None,
                 nodes_hitrates=None,
                 nodes_missing_value_tracks_true=None,
                 nodes_modes=None,
                 nodes_nodeids=None,
                 nodes_treeids=None,
                 nodes_truenodeids=None,
                 nodes_values=None,
                 post_transform="NONE"):
        self.manager = manager
        self.base_values = base_values
        self.class_ids = class_ids
        self.class_nodeids = class_nodeids
        self.class_treeids = class_treeids
        self.class_weights = class_weights
        self.classlabels_int64s = classlabels_int64s
        self.classlabels_strings = classlabels_strings
        self.nodes_falsenodeids = nodes_falsenodeids
        self.nodes_featureids = nodes_featureids
        self.nodes_hitrates = nodes_hitrates
        self.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true
        self.nodes_modes = nodes_modes
        self.nodes_nodeids = nodes_nodeids
        self.nodes_treeids = nodes_treeids
        self.nodes_truenodeids = nodes_truenodeids
        self.nodes_values = nodes_values
        self.post_transform = post_transform

        # Parse tree structure and compile shader (done later or if attributes set)
        if nodes_treeids is not None:
            self.shader_source = self.generate_shader()
            self.classifier_shader = compile_source(self.shader_source)

    def set_attributes(self,
                       base_values=None,
                       class_ids=None,
                       class_nodeids=None,
                       class_treeids=None,
                       class_weights=None,
                       classlabels_int64s=None,
                       classlabels_strings=None,
                       nodes_falsenodeids=None,
                       nodes_featureids=None,
                       nodes_hitrates=None,
                       nodes_missing_value_tracks_true=None,
                       nodes_modes=None,
                       nodes_nodeids=None,
                       nodes_treeids=None,
                       nodes_truenodeids=None,
                       nodes_values=None,
                       post_transform=None):
        if base_values is not None: self.base_values = base_values
        if class_ids is not None: self.class_ids = class_ids
        if class_nodeids is not None: self.class_nodeids = class_nodeids
        if class_treeids is not None: self.class_treeids = class_treeids
        if class_weights is not None: self.class_weights = class_weights
        if classlabels_int64s is not None: self.classlabels_int64s = classlabels_int64s
        if classlabels_strings is not None: self.classlabels_strings = classlabels_strings
        if nodes_falsenodeids is not None: self.nodes_falsenodeids = nodes_falsenodeids
        if nodes_featureids is not None: self.nodes_featureids = nodes_featureids
        if nodes_hitrates is not None: self.nodes_hitrates = nodes_hitrates
        if nodes_missing_value_tracks_true is not None: self.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true
        if nodes_modes is not None: self.nodes_modes = nodes_modes
        if nodes_nodeids is not None: self.nodes_nodeids = nodes_nodeids
        if nodes_treeids is not None: self.nodes_treeids = nodes_treeids
        if nodes_truenodeids is not None: self.nodes_truenodeids = nodes_truenodeids
        if nodes_values is not None: self.nodes_values = nodes_values
        if post_transform is not None: self.post_transform = post_transform

        self.shader_source = self.generate_shader()
        self.classifier_shader = compile_source(self.shader_source)

    def generate_shader(self):
        # Parse tree structure
        nodes_treeids = self.nodes_treeids if self.nodes_treeids else []
        nodes_nodeids = self.nodes_nodeids if self.nodes_nodeids else []
        nodes_featureids = self.nodes_featureids if self.nodes_featureids else []
        nodes_values = self.nodes_values if self.nodes_values else []
        nodes_modes = self.nodes_modes if self.nodes_modes else []
        nodes_truenodeids = self.nodes_truenodeids if self.nodes_truenodeids else []
        nodes_falsenodeids = self.nodes_falsenodeids if self.nodes_falsenodeids else []
        nodes_missing_value_tracks_true = self.nodes_missing_value_tracks_true if self.nodes_missing_value_tracks_true else []

        class_treeids = self.class_treeids if self.class_treeids else []
        class_nodeids = self.class_nodeids if self.class_nodeids else []
        class_ids = self.class_ids if self.class_ids else []
        class_weights = self.class_weights if self.class_weights else []
        base_values = self.base_values if self.base_values else []

        # Determine num_classes
        n_classes_from_labels = max(len(self.classlabels_int64s or []), len(self.classlabels_strings or []))
        n_classes_from_ids = max(class_ids) + 1 if class_ids else 0
        num_classes = max(n_classes_from_labels, n_classes_from_ids, 2)

        # Build tree map: tree_id -> {node_id -> node_info}
        trees = {}
        for i in range(len(nodes_treeids)):
            tid = nodes_treeids[i]
            nid = nodes_nodeids[i]
            if tid not in trees:
                trees[tid] = {}

            mode = nodes_modes[i]
            if isinstance(mode, bytes):
                mode = mode.decode('utf-8')

            trees[tid][nid] = {
                'feature_id': nodes_featureids[i],
                'value': nodes_values[i],
                'mode': mode,
                'true_node_id': nodes_truenodeids[i] if i < len(nodes_truenodeids) else 0,
                'false_node_id': nodes_falsenodeids[i] if i < len(nodes_falsenodeids) else 0,
                'missing_tracks_true': nodes_missing_value_tracks_true[i] if i < len(nodes_missing_value_tracks_true) else 0
            }

        # Build leaf weights map: (tree_id, node_id) -> [(class_id, weight), ...]
        leaf_weights = {}
        for i in range(len(class_treeids)):
            tid = class_treeids[i]
            nid = class_nodeids[i]
            key = (tid, nid)
            if key not in leaf_weights:
                leaf_weights[key] = []
            leaf_weights[key].append((class_ids[i], class_weights[i]))

        # Detect binary mode
        unique_class_ids = set(class_ids) if class_ids else set()
        is_binary = (len(unique_class_ids) == 1 and num_classes == 2)
        binary_mode = 0
        if is_binary:
            if self.post_transform in ("NONE", "PROBIT"):
                binary_mode = 1 # scores[0] = 1 - scores[1]
            else:
                binary_mode = 2 # scores[0] = -scores[1]

        # Generate GLSL
        shader = []
        shader.append("#version 450")
        shader.append(f"layout (local_size_x = {LOCAL_X_1D}) in;")
        shader.append("layout(std430, set=0, binding=0) readonly buffer InputX { float X[]; };")
        shader.append("layout(std430, set=0, binding=1) readonly buffer BaseValues { float base_values[]; };")
        shader.append("layout(std430, set=0, binding=2) writeonly buffer OutputScores { float output_scores[]; };")
        shader.append("layout(std430, set=0, binding=3) writeonly buffer OutputLabels { float output_labels[]; };")

        binding_idx = 4
        # Label map buffer if needed
        has_label_map = self.classlabels_int64s is not None and len(self.classlabels_int64s) > 0
        if has_label_map:
             shader.append(f"layout(std430, set=0, binding={binding_idx}) readonly buffer LabelMap {{ float label_map[]; }};")
             binding_idx += 1

        shader.append(f"layout(std430, set=0, binding={binding_idx}) readonly buffer Params {{")
        shader.append("    uint num_samples;")
        shader.append("    uint num_features;")
        shader.append("};")

        shader.append(f"const uint num_classes = {num_classes};")

        shader.append("const float FLT_MAX = 3.402823466e+38;")
        shader.append("const float FLT_MIN = -3.402823466e+38;")

        # Probit function
        if self.post_transform == "PROBIT":
            shader.append("""
float probit_approx(float x) {
    const float sqrt2_inv = 0.70710678;
    float scaled = x * sqrt2_inv;
    float t = 1.0 / (1.0 + 0.5 * abs(scaled));
    float tau = t * exp(-scaled * scaled - 1.26551223 +
                        t * (1.00002368 +
                        t * (0.37409196 +
                        t * (0.09678418 +
                        t * (-0.18628806 +
                        t * (0.27886807 +
                        t * (-1.13520398 +
                        t * (1.48851587 +
                        t * (-0.82215223 +
                        t * 0.17087277)))))))));
    float erf_val = (scaled >= 0.0) ? (1.0 - tau) : (tau - 1.0);
    return 0.5 * (1.0 + erf_val);
}
""")

        shader.append("void main() {")
        shader.append("    uint sample_idx = gl_GlobalInvocationID.x;")
        shader.append("    if (sample_idx >= num_samples) return;")

        shader.append("    float scores[num_classes];")
        # Initialize scores
        for c in range(num_classes):
            # We assume base_values buffer is at least num_classes long and filled appropriately
            shader.append(f"    scores[{c}] = base_values[{c}];")

        shader.append("    uint sample_offset = sample_idx * num_features;")

        # Generate code for each tree
        sorted_tree_ids = sorted(trees.keys())
        for tid in sorted_tree_ids:
            shader.append(f"    // Tree {tid}")
            shader.append("    {")

            def gen_node(nid):
                node = trees[tid][nid]
                mode = node['mode']

                if mode == 'LEAF':
                    weights = leaf_weights.get((tid, nid), [])
                    if not weights:
                        return
                    for cls_id, weight in weights:
                        if cls_id < num_classes:
                            shader.append(f"        scores[{cls_id}] += {weight};")
                else:
                    feat_id = node['feature_id']
                    val = node['value']
                    true_nid = node['true_node_id']
                    false_nid = node['false_node_id']
                    missing_true = node['missing_tracks_true']

                    shader.append(f"        float val_{nid} = X[sample_offset + {feat_id}];")
                    shader.append(f"        bool cond_{nid} = false;")

                    if missing_true:
                        shader.append(f"        if (isnan(val_{nid})) cond_{nid} = true;")
                    else:
                        shader.append(f"        if (isnan(val_{nid})) cond_{nid} = false;")

                    shader.append(f"        else {{")

                    if mode == 'BRANCH_LEQ':
                        shader.append(f"            cond_{nid} = val_{nid} <= {val};")
                    elif mode == 'BRANCH_LT':
                        shader.append(f"            cond_{nid} = val_{nid} < {val};")
                    elif mode == 'BRANCH_GTE':
                        shader.append(f"            cond_{nid} = val_{nid} >= {val};")
                    elif mode == 'BRANCH_GT':
                        shader.append(f"            cond_{nid} = val_{nid} > {val};")
                    elif mode == 'BRANCH_EQ':
                        shader.append(f"            cond_{nid} = val_{nid} == {val};")
                    elif mode == 'BRANCH_NEQ':
                        shader.append(f"            cond_{nid} = val_{nid} != {val};")

                    shader.append("        }")

                    shader.append(f"        if (cond_{nid}) {{")
                    gen_node(true_nid)
                    shader.append("        } else {")
                    gen_node(false_nid)
                    shader.append("        }")

            root_id = min(trees[tid].keys())
            gen_node(root_id)
            shader.append("    }")

        # Binary mode handling
        if binary_mode == 1:
            shader.append("    scores[0] = 1.0 - scores[1];")
        elif binary_mode == 2:
            shader.append("    scores[0] = -scores[1];")

        # Post transform logic (same as before)
        if self.post_transform == "SOFTMAX":
            shader.append("    float v_max = scores[0];")
            shader.append("    for (int i = 1; i < num_classes; ++i) { if (scores[i] > v_max) v_max = scores[i]; }")
            shader.append("    float sum_exp = 0.0;")
            shader.append("    for (int i = 0; i < num_classes; ++i) { scores[i] = exp(scores[i] - v_max); sum_exp += scores[i]; }")
            shader.append("    for (int i = 0; i < num_classes; ++i) { scores[i] /= sum_exp; }")
        elif self.post_transform == "SOFTMAX_ZERO":
            shader.append("    float v_max = scores[0];")
            shader.append("    for (int i = 1; i < num_classes; ++i) { if (scores[i] > v_max) v_max = scores[i]; }")
            shader.append("    float exp_neg_v_max = exp(-v_max);")
            shader.append("    float sum_val = 0.0;")
            shader.append("    for (int i = 0; i < num_classes; ++i) {")
            shader.append("        if (scores[i] > 0.0000001 || scores[i] < -0.0000001) scores[i] = exp(scores[i] - v_max);")
            shader.append("        else scores[i] = scores[i] * exp_neg_v_max;")
            shader.append("        sum_val += scores[i];")
            shader.append("    }")
            shader.append("    float norm_val = (sum_val == 0.0) ? 0.5 : (1.0 / sum_val);")
            shader.append("    for (int i = 0; i < num_classes; ++i) { scores[i] *= norm_val; }")
        elif self.post_transform == "LOGISTIC":
            shader.append("    for (int i = 0; i < num_classes; ++i) {")
            shader.append("        float val = scores[i];")
            shader.append("        float abs_val = abs(val);")
            shader.append("        float v = 1.0 / (1.0 + exp(-abs_val));")
            shader.append("        scores[i] = (val < 0.0) ? (1.0 - v) : v;")
            shader.append("    }")
        elif self.post_transform == "PROBIT":
            shader.append("    for (int i = 0; i < num_classes; ++i) { scores[i] = probit_approx(scores[i]); }")

        # Write scores
        shader.append("    uint out_base = sample_idx * num_classes;")
        shader.append("    for (int i = 0; i < num_classes; ++i) { output_scores[out_base + i] = scores[i]; }")

        # Argmax
        shader.append("    float max_val = scores[0];")
        shader.append("    uint max_idx = 0;")
        shader.append("    for (uint i = 1; i < num_classes; ++i) {")
        shader.append("        if (scores[i] > max_val) { max_val = scores[i]; max_idx = i; }")
        shader.append("    }")

        # Label mapping
        if has_label_map:
            shader.append("    int label_val = floatBitsToInt(label_map[max_idx]);")
            shader.append("    output_labels[sample_idx] = float(label_val);")
        else:
            shader.append("    output_labels[sample_idx] = float(max_idx);")

        shader.append("}")

        return "\n".join(shader)

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"TreeEnsembleClassifierOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        output_tensors = [t[0] for t in output_tensor_and_shape]

        seq = self.manager.sequence()
        all_tensors_to_sync = [t[0] for t in input_tensors] + updated_tensors
        seq.record(kp.OpTensorSyncDevice(all_tensors_to_sync))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal(output_tensors))
        seq.eval()

        outputs = []
        for (tensor_out, output_shape) in output_tensor_and_shape:
            outputs.append(tensor_out.data().reshape(output_shape))

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        X_tensor, X_shape = input_tensors[0]
        n_samples = X_shape[0]
        n_features = int(np.prod(X_shape[1:]))

        # 处理属性的默认值
        base_values = self.base_values if self.base_values is not None else []
        class_ids = self.class_ids if self.class_ids is not None else []

        # 确定类别数
        n_classes_from_labels = max(len(self.classlabels_int64s or []), len(self.classlabels_strings or []))
        n_classes_from_ids = max(class_ids) + 1 if class_ids else 0
        num_classes = max(n_classes_from_labels, n_classes_from_ids, 2)

        # 处理base values
        if not base_values:
            base_values = [0.0] * num_classes

        # 创建GPU buffers
        tensor_base_values = self.manager.tensor(np.array(base_values, dtype=np.float32))
        updated_tensors.append(tensor_base_values)

        # 输出scores
        tensor_scores = self.manager.tensor(np.zeros(n_samples * num_classes, dtype=np.float32))
        updated_tensors.append(tensor_scores)

        # 输出labels
        tensor_labels = self.manager.tensor(np.zeros(n_samples, dtype=np.float32))
        updated_tensors.append(tensor_labels)

        params = [X_tensor, tensor_base_values, tensor_scores, tensor_labels]

        # Label map
        if self.classlabels_int64s is not None and len(self.classlabels_int64s) > 0:
            if len(self.classlabels_int64s) == 1 and self.classlabels_int64s[0] == 1:
                label_map_data = np.array([0, 1], dtype=np.int32).view(np.float32)
            else:
                label_map_data = np.array(self.classlabels_int64s, dtype=np.int32).view(np.float32)
            tensor_label_map = self.manager.tensor(label_map_data)
            updated_tensors.append(tensor_label_map)
            params.append(tensor_label_map)

        # Params buffer: num_samples, num_features
        params_array = np.array([n_samples, n_features], dtype=np.uint32)
        params_tensor = self.manager.tensor_t(params_array, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([params_tensor])).eval()
        updated_tensors.append(params_tensor)
        params.append(params_tensor)

        # Step 1: 主分类
        workgroup = ((n_samples + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

        updated_algorithms.append(self.manager.algorithm(
            params,
            self.classifier_shader,
            workgroup
        ))

        return [
            (tensor_labels, [n_samples]),
            (tensor_scores, [n_samples, num_classes])
        ]

