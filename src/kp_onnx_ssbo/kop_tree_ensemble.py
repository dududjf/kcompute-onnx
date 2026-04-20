import kp
import numpy as np
from .shader_utils import compile_source, LOCAL_X_1D


class TreeEnsembleOp:
    def __init__(self,
                 manager: kp.Manager,
                 nodes_splits=None,
                 nodes_featureids=None,
                 nodes_modes=None,
                 nodes_truenodeids=None,
                 nodes_falsenodeids=None,
                 nodes_trueleafs=None,
                 nodes_falseleafs=None,
                 leaf_targetids=None,
                 leaf_weights=None,
                 tree_roots=None,
                 n_targets=None,
                 aggregate_function="SUM",
                 post_transform="NONE",
                 nodes_missing_value_tracks_true=None,
                 membership_values=None):
        self.manager = manager
        assert post_transform in [None, "NONE"], f"post_transform must be NONE, got {post_transform}"
        self.nodes_splits = nodes_splits if nodes_splits is not None else []
        self.nodes_featureids = nodes_featureids if nodes_featureids is not None else []
        self.nodes_modes = nodes_modes if nodes_modes is not None else []
        self.nodes_truenodeids = nodes_truenodeids if nodes_truenodeids is not None else []
        self.nodes_falsenodeids = nodes_falsenodeids if nodes_falsenodeids is not None else []
        self.nodes_trueleafs = nodes_trueleafs if nodes_trueleafs is not None else []
        self.nodes_falseleafs = nodes_falseleafs if nodes_falseleafs is not None else []
        self.leaf_targetids = leaf_targetids if leaf_targetids is not None else []
        self.leaf_weights = leaf_weights if leaf_weights is not None else []
        self.tree_roots = tree_roots if tree_roots is not None else []
        self.n_targets = n_targets if n_targets is not None else 1
        self.aggregate_function = aggregate_function
        self.post_transform = post_transform
        self.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true if nodes_missing_value_tracks_true is not None else []
        self.membership_values = membership_values

        self.shader_source = self.generate_shader()
        self.program = compile_source(self.shader_source)

    def set_attributes(self,
                       nodes_splits=None,
                       nodes_featureids=None,
                       nodes_modes=None,
                       nodes_truenodeids=None,
                       nodes_falsenodeids=None,
                       nodes_trueleafs=None,
                       nodes_falseleafs=None,
                       leaf_targetids=None,
                       leaf_weights=None,
                       tree_roots=None,
                       n_targets=None,
                       aggregate_function=None,
                       post_transform=None,
                       nodes_missing_value_tracks_true=None,
                       membership_values=None):
        if nodes_splits is not None: self.nodes_splits = nodes_splits
        if nodes_featureids is not None: self.nodes_featureids = nodes_featureids
        if nodes_modes is not None: self.nodes_modes = nodes_modes
        if nodes_truenodeids is not None: self.nodes_truenodeids = nodes_truenodeids
        if nodes_falsenodeids is not None: self.nodes_falsenodeids = nodes_falsenodeids
        if nodes_trueleafs is not None: self.nodes_trueleafs = nodes_trueleafs
        if nodes_falseleafs is not None: self.nodes_falseleafs = nodes_falseleafs
        if leaf_targetids is not None: self.leaf_targetids = leaf_targetids
        if leaf_weights is not None: self.leaf_weights = leaf_weights
        if tree_roots is not None: self.tree_roots = tree_roots
        if n_targets is not None: self.n_targets = n_targets
        if aggregate_function is not None: self.aggregate_function = aggregate_function
        if post_transform is not None: self.post_transform = post_transform
        if nodes_missing_value_tracks_true is not None: self.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true
        if membership_values is not None: self.membership_values = membership_values

        self.shader_source = self.generate_shader()
        self.program = compile_source(self.shader_source)

    def generate_shader(self):
        # We assume n_targets is small enough or we handle it by dispatching logic.
        # Original dispatch was (n_samples, n_targets, 1).
        # In SSBO we can do 1D dispatch over (n_samples * n_targets) or just over n_samples and loop targets.
        # Since n_targets is property of model (not input shape), we can loop inside shader or dispatch 2D.
        # Let's stick to 2D dispatch logic (x=samples, y=targets) but map linear thread ID if we want,
        # or just use proper local_size.

        # Let's use local_size_x=LOCAL_X_1D, and flatten indices.

        shader = []
        shader.append("#version 450")
        shader.append(f"layout(local_size_x={LOCAL_X_1D}) in;")
        shader.append("layout(std430, set=0, binding=0) readonly buffer InputX { float X[]; };")
        shader.append("layout(std430, set=0, binding=1) writeonly buffer OutputScores { float output_scores[]; };")
        shader.append("layout(std430, set=0, binding=2) readonly buffer Params {")
        shader.append("    uint num_samples;")
        shader.append("    uint num_features;")
        shader.append("    uint num_targets;")
        shader.append("    uint agg_mode;")
        shader.append("};")

        shader.append("const float FLT_MAX = 3.402823466e+38;")
        shader.append("const float FLT_MIN = -3.402823466e+38;")

        shader.append("void main() {")
        shader.append("    uint idx = gl_GlobalInvocationID.x;")
        shader.append("    if (idx >= num_samples * num_targets) return;")

        shader.append("    uint sample_idx = idx / num_targets;")
        shader.append("    uint target_idx = idx - sample_idx * num_targets;")

        shader.append("    float score;")
        shader.append("    if (agg_mode == 0 || agg_mode == 3) { score = 0.0; }")
        shader.append("    else if (agg_mode == 1) { score = FLT_MAX; }")
        shader.append("    else { score = FLT_MIN; }")

        shader.append("    uint sample_offset = sample_idx * num_features;")

        self.membership_iter_idx = 0

        # Helper to generate node code
        def gen_node(nid, is_leaf):
            if is_leaf:
                target_id = self.leaf_targetids[nid]
                weight = self.leaf_weights[nid]

                shader.append(f"        if (target_idx == {target_id}) {{")
                shader.append(f"            float w = {weight};")
                shader.append("            if (agg_mode == 0 || agg_mode == 3) score += w;")
                shader.append("            else if (agg_mode == 1) score = min(score, w);")
                shader.append("            else score = max(score, w);")
                shader.append("        }")
                return

            feat_id = self.nodes_featureids[nid]
            val = self.nodes_splits[nid]
            mode = self.nodes_modes[nid]
            missing_true = self.nodes_missing_value_tracks_true[nid] if nid < len(self.nodes_missing_value_tracks_true) else 0

            true_child = self.nodes_truenodeids[nid]
            false_child = self.nodes_falsenodeids[nid]
            true_is_leaf = self.nodes_trueleafs[nid]
            false_is_leaf = self.nodes_falseleafs[nid]

            shader.append(f"        float val_{nid} = X[sample_offset + {feat_id}];")
            shader.append(f"        bool cond_{nid} = false;")

            if missing_true:
                shader.append(f"        if (isnan(val_{nid})) cond_{nid} = true;")
            else:
                shader.append(f"        if (isnan(val_{nid})) cond_{nid} = false;")

            shader.append(f"        else {{")

            # Modes: 0: LEQ, 1: LT, 2: GTE, 3: GT, 4: EQ, 5: NEQ
            if mode == 0: # LEQ
                shader.append(f"            cond_{nid} = val_{nid} <= {val};")
            elif mode == 1: # LT
                shader.append(f"            cond_{nid} = val_{nid} < {val};")
            elif mode == 2: # GTE
                shader.append(f"            cond_{nid} = val_{nid} >= {val};")
            elif mode == 3: # GT
                shader.append(f"            cond_{nid} = val_{nid} > {val};")
            elif mode == 4: # EQ
                shader.append(f"            cond_{nid} = val_{nid} == {val};")
            elif mode == 5: # NEQ
                shader.append(f"            cond_{nid} = val_{nid} != {val};")
            elif mode == 6: # MEMBER
                members = []
                if self.membership_values is not None:
                    while self.membership_iter_idx < len(self.membership_values):
                        m_val = self.membership_values[self.membership_iter_idx]
                        self.membership_iter_idx += 1
                        if np.isnan(m_val):
                            break
                        members.append(m_val)

                if not members:
                    shader.append(f"            cond_{nid} = false;")
                else:
                    checks = [f"val_{nid} == {m}" for m in members]
                    shader.append(f"            cond_{nid} = {' || '.join(checks)};")

            shader.append("        }")

            shader.append(f"        if (cond_{nid}) {{")
            gen_node(true_child, true_is_leaf)
            shader.append("        } else {")
            gen_node(false_child, false_is_leaf)
            shader.append("        }")

        # Generate code for each tree
        for i, root_id in enumerate(self.tree_roots):
            shader.append(f"    // Tree {i}")
            shader.append("    {")

            is_root_leaf = False
            if root_id < len(self.nodes_trueleafs):
                 if (self.nodes_trueleafs[root_id] and self.nodes_falseleafs[root_id] and
                     self.nodes_truenodeids[root_id] == self.nodes_falsenodeids[root_id]):
                     is_root_leaf = True

            if is_root_leaf:
                leaf_idx = self.nodes_truenodeids[root_id]
                gen_node(leaf_idx, True)
            else:
                gen_node(root_id, False)

            shader.append("    }")

        if self.aggregate_function == "AVERAGE":
             shader.append(f"    score /= float({len(self.tree_roots)});")

        shader.append("    output_scores[idx] = score;")
        shader.append("}")

        return "\n".join(shader)

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"TreeEnsembleOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensors_and_shapes = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        seq = self.manager.sequence()
        all_tensors = [t[0] for t in input_tensors] + updated_tensors
        seq.record(kp.OpTensorSyncDevice(all_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        output_tensors = [t[0] for t in output_tensors_and_shapes]
        seq.record(kp.OpTensorSyncLocal(output_tensors))
        seq.eval()

        results = []
        for i, (tensor, shape) in enumerate(output_tensors_and_shapes):
            data = tensor.data()
            if shape:
                data = data.reshape(shape)
            results.append(data)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors

        return results

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        X_tensor, X_shape = input_tensors[0]
        n_samples = X_shape[0]
        n_features = int(np.prod(X_shape[1:]))

        # Aggregation mode
        agg_mode_map = {"SUM": 0, "MIN": 1, "MAX": 2, "AVERAGE": 3}
        agg_mode = agg_mode_map.get(self.aggregate_function, 0)

        # Output scores
        tensor_scores = self.manager.tensor(np.zeros(n_samples * self.n_targets, dtype=np.float32))
        updated_tensors.append(tensor_scores)

        # Params: num_samples, num_features, num_targets, agg_mode
        params = np.array([n_samples, n_features, self.n_targets, agg_mode], dtype=np.uint32)
        params_tensor = self.manager.tensor_t(params, kp.TensorTypes.device)
        self.manager.sequence().record(kp.OpTensorSyncDevice([params_tensor])).eval()
        updated_tensors.append(params_tensor)

        total_threads = n_samples * self.n_targets
        workgroup = ((total_threads + LOCAL_X_1D - 1) // LOCAL_X_1D, 1, 1)

        # Dispatch
        updated_algorithms.append(self.manager.algorithm(
            [X_tensor, tensor_scores, params_tensor],
            self.program,
            workgroup
        ))

        return [
            (tensor_scores, [n_samples, self.n_targets])
        ]

