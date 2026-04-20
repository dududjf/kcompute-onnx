import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kp
import numpy as np
import time
from kp_onnx_ssbo.kop_tree_ensemble_classifier import TreeEnsembleClassifierOp


# --------------------------
# ONNX TreeEnsembleClassifier baseline (copied from op_tree_ensemble_classifier.py)
# --------------------------

def logistic(x):
    return 1 / (1 + np.exp(-x))


def probit(x):
    from scipy.special import ndtr
    return ndtr(x)


def softmax(x):
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    mx = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - mx)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_zero(x):
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    mx = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - mx)
    s = np.sum(exp_x, axis=1, keepdims=True)
    res = exp_x / s
    res[s.squeeze() == 0] = 0
    return res


class TreeAttributes:
    def __init__(self):
        pass


class TreeEnsemble:
    def __init__(
        self,
        base_values=None,
        base_values_as_tensor=None,
        nodes_falsenodeids=None,
        nodes_featureids=None,
        nodes_hitrates=None,
        nodes_hitrates_as_tensor=None,
        nodes_missing_value_tracks_true=None,
        nodes_modes=None,
        nodes_nodeids=None,
        nodes_treeids=None,
        nodes_truenodeids=None,
        nodes_values=None,
        nodes_values_as_tensor=None,
        class_weights=None,
        class_weights_as_tensor=None,
    ):
        self.atts = TreeAttributes()
        self.atts.base_values = base_values
        self.atts.base_values_as_tensor = base_values_as_tensor
        self.atts.nodes_falsenodeids = nodes_falsenodeids
        self.atts.nodes_featureids = nodes_featureids
        self.atts.nodes_hitrates = nodes_hitrates
        self.atts.nodes_hitrates_as_tensor = nodes_hitrates_as_tensor
        self.atts.nodes_missing_value_tracks_true = nodes_missing_value_tracks_true
        self.atts.nodes_modes = nodes_modes
        self.atts.nodes_nodeids = nodes_nodeids
        self.atts.nodes_treeids = nodes_treeids
        self.atts.nodes_truenodeids = nodes_truenodeids
        self.atts.nodes_values = nodes_values
        self.atts.nodes_values_as_tensor = nodes_values_as_tensor
        self.atts.class_weights = class_weights
        self.atts.class_weights_as_tensor = class_weights_as_tensor

    def leave_index_tree(self, X):
        leaves = []
        for i in range(X.shape[0]):
            leave = self._find_leaves_for_sample(X[i])
            leaves.append(leave)
        return np.array(leaves)

    def _find_leaves_for_sample(self, x):
        trees = {}
        for i, tid in enumerate(self.atts.nodes_treeids):
            if tid not in trees:
                trees[tid] = []
            trees[tid].append(i)

        leaves = []
        for tid in sorted(trees.keys()):
            node_idx = trees[tid][0]
            while True:
                mode = self.atts.nodes_modes[node_idx]
                if mode == "LEAF":
                    leaves.append(node_idx)
                    break

                fid = self.atts.nodes_featureids[node_idx]
                val = x[fid]
                threshold = self.atts.nodes_values[node_idx]

                if np.isnan(val):
                    if self.atts.nodes_missing_value_tracks_true and self.atts.nodes_missing_value_tracks_true[node_idx]:
                        go_true = True
                    else:
                        go_true = False
                else:
                    if mode == "BRANCH_LEQ":
                        go_true = val <= threshold
                    elif mode == "BRANCH_LT":
                        go_true = val < threshold
                    elif mode == "BRANCH_GTE":
                        go_true = val >= threshold
                    elif mode == "BRANCH_GT":
                        go_true = val > threshold
                    elif mode == "BRANCH_EQ":
                        go_true = val == threshold
                    elif mode == "BRANCH_NEQ":
                        go_true = val != threshold
                    else:
                        go_true = False

                next_nid = self.atts.nodes_truenodeids[node_idx] if go_true else self.atts.nodes_falsenodeids[node_idx]

                # Find next node
                found = False
                for i in trees[tid]:
                    if self.atts.nodes_nodeids[i] == next_nid:
                        node_idx = i
                        found = True
                        break

                if not found:
                    leaves.append(node_idx)
                    break

        return leaves


def onnx_tree_ensemble_classifier(
    X,
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
    post_transform=None,
):
    tr = TreeEnsemble(
        base_values=base_values,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_featureids=nodes_featureids,
        nodes_hitrates=nodes_hitrates,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        nodes_modes=nodes_modes,
        nodes_nodeids=nodes_nodeids,
        nodes_treeids=nodes_treeids,
        nodes_truenodeids=nodes_truenodeids,
        nodes_values=nodes_values,
        class_weights=class_weights,
    )

    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float32)

    leaves_index = tr.leave_index_tree(X)
    n_classes_labels = max(len(classlabels_int64s or []), len(classlabels_strings or []))
    n_classes_ids = max(class_ids) + 1 if class_ids else 0
    n_classes = max(n_classes_labels, n_classes_ids)
    if n_classes == 0: n_classes = 2 # Default to binary if nothing specified

    res = np.empty((leaves_index.shape[0], n_classes), dtype=np.float32)

    if tr.atts.base_values is None:
        res[:, :] = 0
    else:
        res[:, :] = np.array(tr.atts.base_values).reshape((1, -1))

    class_index = {}
    for i, (tid, nid) in enumerate(zip(class_treeids, class_nodeids)):
        if (tid, nid) not in class_index:
            class_index[tid, nid] = []
        class_index[tid, nid].append(i)

    for i in range(res.shape[0]):
        indices = leaves_index[i]
        t_index = [class_index.get((nodes_treeids[idx], nodes_nodeids[idx]), []) for idx in indices]
        for its in t_index:
            for it in its:
                res[i, class_ids[it]] += tr.atts.class_weights[it]

    # post_transform
    binary = len(set(class_ids)) == 1
    classes = classlabels_int64s or classlabels_strings
    post_function = {
        None: lambda x: x,
        "NONE": lambda x: x,
        "LOGISTIC": logistic,
        "SOFTMAX": softmax,
        "SOFTMAX_ZERO": softmax_zero,
        "PROBIT": probit,
    }

    if binary:
        if res.shape[1] == len(classes) == 1:
            new_res = np.zeros((res.shape[0], 2), res.dtype)
            new_res[:, 1] = res[:, 0]
            res = new_res
        else:
            if not (class_ids is not None and 1 in class_ids and 0 not in class_ids):
                res[:, 1] = res[:, 0]
        if post_transform in (None, "NONE", "PROBIT"):
            res[:, 0] = 1 - res[:, 1]
        else:
            res[:, 0] = -res[:, 1]

    new_scores = post_function[post_transform](res)
    labels = np.argmax(new_scores, axis=1)

    # labels
    if classlabels_int64s is not None:
        if len(classlabels_int64s) == 1:
            if classlabels_int64s[0] == 1:
                d = {1: 1}
                labels = np.array([d.get(i, 0) for i in labels], dtype=np.int64)
            else:
                raise NotImplementedError(f"classlabels_int64s={classlabels_int64s}, not supported.")
        else:
            labels = np.array([classlabels_int64s[i] for i in labels], dtype=np.int64)
    elif classlabels_strings is not None:
        if len(classlabels_strings) == 1:
            raise NotImplementedError(f"classlabels_strings={classlabels_strings}, not supported.")
        labels = np.array([classlabels_strings[i] for i in labels])

    return labels, new_scores


# --------------------------
# Test Cases
# --------------------------

mgr = kp.Manager()

# Case 1: Binary Classification with SOFTMAX
print("Case 1: Binary Classification with SOFTMAX")
params1 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [1.0, 1.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "SOFTMAX"
}

x1 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels, onnx_scores = onnx_tree_ensemble_classifier(x1, **params1)
print("ONNX:", onnx_scores.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op1 = TreeEnsembleClassifierOp(mgr, **params1)
kp_labels, kp_scores = tree_op1.run(x1)
print(f"{tree_op1}:", kp_scores.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores.shape == onnx_scores.shape)
print("Scores Max error:", np.abs(onnx_scores - kp_scores).max())
print("Scores All close:", np.allclose(onnx_scores, kp_scores, rtol=1e-4, atol=1e-4))
print("Labels shape equal:", kp_labels.shape == onnx_labels.shape)
print("Labels Max error:", np.abs(onnx_labels - kp_labels).max())
print("Labels All close:", np.allclose(onnx_labels, kp_labels))
print("----")

# Case 2: Multi-class Classification (3 classes)
print("Case 2: Multi-class Classification (3 classes) with SOFTMAX")
params2 = {
    "nodes_falsenodeids": [2, 4, 0, 0, 0],
    "nodes_featureids": [0, 1, 0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "BRANCH_LT", "LEAF", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2, 3, 4],
    "nodes_treeids": [0, 0, 0, 0, 0],
    "nodes_truenodeids": [1, 3, 0, 0, 0],
    "nodes_values": [0.5, 0.5, 0.0, 0.0, 0.0],
    "class_ids": [0, 1, 2],
    "class_nodeids": [2, 3, 4],
    "class_treeids": [0, 0, 0],
    "class_weights": [10.0, 20.0, 30.0],
    "classlabels_int64s": [0, 1, 2],
    "post_transform": "SOFTMAX"
}

x2 = np.random.rand(200, 2).astype(np.float32)

t0 = time.time()
onnx_labels2, onnx_scores2 = onnx_tree_ensemble_classifier(x2, **params2)
print("ONNX:", onnx_scores2.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op2 = TreeEnsembleClassifierOp(mgr, **params2)
kp_labels2, kp_scores2 = tree_op2.run(x2)
print(f"{tree_op2}:", kp_scores2.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores2.shape == onnx_scores2.shape)
print("Scores Max error:", np.abs(onnx_scores2 - kp_scores2).max())
print("Scores All close:", np.allclose(onnx_scores2, kp_scores2, rtol=1e-4, atol=1e-4))
print("Labels shape equal:", kp_labels2.shape == onnx_labels2.shape)
print("Labels Max error:", np.abs(onnx_labels2 - kp_labels2).max())
print("Labels All close:", np.allclose(onnx_labels2, kp_labels2))
print("----")

# Case 3: NONE post_transform
print("Case 3: Binary Classification with NONE post_transform")
params3 = dict(params1)
params3["post_transform"] = "NONE"

x3 = np.random.rand(200, 1).astype(np.float32)

t0 = time.time()
onnx_labels3, onnx_scores3 = onnx_tree_ensemble_classifier(x3, **params3)
print("ONNX:", onnx_scores3.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op3 = TreeEnsembleClassifierOp(mgr, **params3)
kp_labels3, kp_scores3 = tree_op3.run(x3)
print(f"{tree_op3}:", kp_scores3.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores3.shape == onnx_scores3.shape)
print("Scores Max error:", np.abs(onnx_scores3 - kp_scores3).max())
print("Scores All close:", np.allclose(onnx_scores3, kp_scores3, rtol=1e-4, atol=1e-4))
print("Labels shape equal:", kp_labels3.shape == onnx_labels3.shape)
print("Labels All close:", np.allclose(onnx_labels3, kp_labels3))
print("----")

# Case 4: LOGISTIC post_transform
print("Case 4: Multi-class with LOGISTIC post_transform")
params4 = dict(params2)
params4["post_transform"] = "LOGISTIC"

x4 = np.random.rand(1000, 2).astype(np.float32)

t0 = time.time()
onnx_labels4, onnx_scores4 = onnx_tree_ensemble_classifier(x4, **params4)
print("ONNX:", onnx_scores4.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op4 = TreeEnsembleClassifierOp(mgr, **params4)
kp_labels4, kp_scores4 = tree_op4.run(x4)
print(f"{tree_op4}:", kp_scores4.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores4.shape == onnx_scores4.shape)
print("Scores Max error:", np.abs(onnx_scores4 - kp_scores4).max())
print("Scores All close:", np.allclose(onnx_scores4, kp_scores4, rtol=1e-4, atol=1e-4))
print("Labels shape equal:", kp_labels4.shape == onnx_labels4.shape)
print("Labels All close:", np.allclose(onnx_labels4, kp_labels4))
print("----")

# Case 5: Multiple trees
print("Case 5: Multiple trees (ensemble)")
params5 = {
    "nodes_falsenodeids": [2, 0, 0, 2, 0, 0],
    "nodes_featureids": [0, 0, 0, 0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0, 0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF", "BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2, 0, 1, 2],
    "nodes_treeids": [0, 0, 0, 1, 1, 1],
    "nodes_truenodeids": [1, 0, 0, 1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
    "class_ids": [0, 1, 0, 1],
    "class_nodeids": [1, 2, 1, 2],
    "class_treeids": [0, 0, 1, 1],
    "class_weights": [5.0, 10.0, 3.0, 7.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "SOFTMAX"
}

x5 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels5, onnx_scores5 = onnx_tree_ensemble_classifier(x5, **params5)
print("ONNX:", onnx_scores5.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op5 = TreeEnsembleClassifierOp(mgr, **params5)
kp_labels5, kp_scores5 = tree_op5.run(x5)
print(f"{tree_op5}:", kp_scores5.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores5.shape == onnx_scores5.shape)
print("Scores Max error:", np.abs(onnx_scores5 - kp_scores5).max())
print("Scores All close:", np.allclose(onnx_scores5, kp_scores5, rtol=1e-4, atol=1e-4))
print("Labels shape equal:", kp_labels5.shape == onnx_labels5.shape)
print("Labels Max error:", np.abs(onnx_labels5 - kp_labels5).max())
print("Labels All close:", np.allclose(onnx_labels5, kp_labels5))
print("----")

# Case 7: SOFTMAX_ZERO post_transform
print("Case 7: SOFTMAX_ZERO post_transform")
params7 = dict(params2)
params7["post_transform"] = "SOFTMAX_ZERO"

x7 = np.random.rand(1000, 2).astype(np.float32)

t0 = time.time()
onnx_labels7, onnx_scores7 = onnx_tree_ensemble_classifier(x7, **params7)
print("ONNX:", onnx_scores7.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op7 = TreeEnsembleClassifierOp(mgr, **params7)
kp_labels7, kp_scores7 = tree_op7.run(x7)
print(f"{tree_op7}:", kp_scores7.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores7.shape == onnx_scores7.shape)
print("Scores Max error:", np.abs(onnx_scores7 - kp_scores7).max())
print("Scores All close:", np.allclose(onnx_scores7, kp_scores7, rtol=1e-4, atol=1e-4))
print("Labels shape equal:", kp_labels7.shape == onnx_labels7.shape)
print("Labels All close:", np.allclose(onnx_labels7, kp_labels7))
print("----")

# Case 8: PROBIT post_transform
print("Case 8: PROBIT post_transform")
params8 = dict(params2)
params8["post_transform"] = "PROBIT"

x8 = np.random.rand(1000, 2).astype(np.float32)

t0 = time.time()
onnx_labels8, onnx_scores8 = onnx_tree_ensemble_classifier(x8, **params8)
print("ONNX:", onnx_scores8.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op8 = TreeEnsembleClassifierOp(mgr, **params8)
kp_labels8, kp_scores8 = tree_op8.run(x8)
print(f"{tree_op8}:", kp_scores8.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores8.shape == onnx_scores8.shape)
print("Scores Max error:", np.abs(onnx_scores8 - kp_scores8).max())
print("Scores All close:", np.allclose(onnx_scores8, kp_scores8, rtol=1e-2, atol=1e-2))
print("Labels shape equal:", kp_labels8.shape == onnx_labels8.shape)
print("Labels All close:", np.allclose(onnx_labels8, kp_labels8))
print("----")

# Case 9: All comparison modes (BRANCH_LEQ, GTE, GT, EQ, NEQ)
print("Case 9: Various comparison modes (BRANCH_LEQ, GTE, GT)")
params9 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_GTE", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [1.0, 1.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "SOFTMAX"
}

x9 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels9, onnx_scores9 = onnx_tree_ensemble_classifier(x9, **params9)
print("ONNX:", onnx_scores9.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op9 = TreeEnsembleClassifierOp(mgr, **params9)
kp_labels9, kp_scores9 = tree_op9.run(x9)
print(f"{tree_op9}:", kp_scores9.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores9.shape == onnx_scores9.shape)
print("Scores Max error:", np.abs(onnx_scores9 - kp_scores9).max())
print("Scores All close:", np.allclose(onnx_scores9, kp_scores9, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels9, kp_labels9))
print("----")

# Case 10: No classlabels (label mapping identity)
print("Case 10: No classlabels_int64s (identity mapping)")
params10 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [1.0, 1.0],
    "post_transform": "SOFTMAX"
}

x10 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels10, onnx_scores10 = onnx_tree_ensemble_classifier(x10, **params10)
print("ONNX:", onnx_scores10.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op10 = TreeEnsembleClassifierOp(mgr, **params10)
kp_labels10, kp_scores10 = tree_op10.run(x10)
print(f"{tree_op10}:", kp_scores10.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores10.shape == onnx_scores10.shape)
print("Scores Max error:", np.abs(onnx_scores10 - kp_scores10).max())
print("Scores All close:", np.allclose(onnx_scores10, kp_scores10, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels10, kp_labels10))
print("----")

# Case 11: classlabels_int64s = [1] special case
print("Case 11: classlabels_int64s=[1] special case")
params11 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [1.0, 1.0],
    "classlabels_int64s": [1],
    "post_transform": "SOFTMAX"
}

x11 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels11, onnx_scores11 = onnx_tree_ensemble_classifier(x11, **params11)
print("ONNX:", onnx_scores11.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op11 = TreeEnsembleClassifierOp(mgr, **params11)
kp_labels11, kp_scores11 = tree_op11.run(x11)
print(f"{tree_op11}:", kp_scores11.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores11.shape == onnx_scores11.shape)
print("Scores Max error:", np.abs(onnx_scores11 - kp_scores11).max())
print("Scores All close:", np.allclose(onnx_scores11, kp_scores11, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels11, kp_labels11))
print("----")

# Case 12: Binary with LOGISTIC (binary_mode=2)
print("Case 12: Binary classification with LOGISTIC (binary_mode=2)")
params12 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [1],
    "class_nodeids": [1],
    "class_treeids": [0],
    "class_weights": [1.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "LOGISTIC"
}

x12 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels12, onnx_scores12 = onnx_tree_ensemble_classifier(x12, **params12)
print("ONNX:", onnx_scores12.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op12 = TreeEnsembleClassifierOp(mgr, **params12)
kp_labels12, kp_scores12 = tree_op12.run(x12)
print(f"{tree_op12}:", kp_scores12.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores12.shape == onnx_scores12.shape)
print("Scores Max error:", np.abs(onnx_scores12 - kp_scores12).max())
print("Scores All close:", np.allclose(onnx_scores12, kp_scores12, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels12, kp_labels12))
print("----")

# Case 13: Comparison modes EQ and NEQ
print("Case 13: Comparison modes EQ and NEQ")
params13 = {
    "nodes_falsenodeids": [2, 0, 0, 2, 0, 0],
    "nodes_featureids": [0, 0, 0, 0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0, 0, 0, 0],
    "nodes_modes": ["BRANCH_EQ", "LEAF", "LEAF", "BRANCH_NEQ", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2, 0, 1, 2],
    "nodes_treeids": [0, 0, 0, 1, 1, 1],
    "nodes_truenodeids": [1, 0, 0, 1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0, 0.75, 0.0, 0.0],
    "class_ids": [0, 1, 0, 1],
    "class_nodeids": [1, 2, 1, 2],
    "class_treeids": [0, 0, 1, 1],
    "class_weights": [10.0, 20.0, 5.0, 15.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "NONE"
}

x13 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels13, onnx_scores13 = onnx_tree_ensemble_classifier(x13, **params13)
print("ONNX:", onnx_scores13.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op13 = TreeEnsembleClassifierOp(mgr, **params13)
kp_labels13, kp_scores13 = tree_op13.run(x13)
print(f"{tree_op13}:", kp_scores13.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores13.shape == onnx_scores13.shape)
print("Scores Max error:", np.abs(onnx_scores13 - kp_scores13).max())
print("Scores All close:", np.allclose(onnx_scores13, kp_scores13, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels13, kp_labels13))
print("----")

# Case 14: Missing values (NaN)
print("Case 14: Missing values (NaN)")
params14 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [1, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [10.0, 20.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "NONE"
}

x14_rand = np.random.rand(1000, 1).astype(np.float32)
x14_nan = np.full((1, 1), np.nan, dtype=np.float32)
x14 = np.vstack([x14_nan, x14_rand[:-1]])

t0 = time.time()
onnx_labels14, onnx_scores14 = onnx_tree_ensemble_classifier(x14, **params14)
print("ONNX:", onnx_scores14.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op14 = TreeEnsembleClassifierOp(mgr, **params14)
kp_labels14, kp_scores14 = tree_op14.run(x14)
print(f"{tree_op14}:", kp_scores14.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores14.shape == onnx_scores14.shape)
print("Scores Max error:", np.abs(onnx_scores14 - kp_scores14).max())
print("Scores All close:", np.allclose(onnx_scores14, kp_scores14, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels14, kp_labels14))
print("----")

# Case 15: Base values
print("Case 15: Base values")
params15 = dict(params1)
params15["base_values"] = [100.0, 200.0]

x15 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels15, onnx_scores15 = onnx_tree_ensemble_classifier(x15, **params15)
print("ONNX:", onnx_scores15.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op15 = TreeEnsembleClassifierOp(mgr, **params15)
kp_labels15, kp_scores15 = tree_op15.run(x15)
print(f"{tree_op15}:", kp_scores15.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores15.shape == onnx_scores15.shape)
print("Scores Max error:", np.abs(onnx_scores15 - kp_scores15).max())
print("Scores All close:", np.allclose(onnx_scores15, kp_scores15, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels15, kp_labels15))
print("----")

# Case 16: Binary mode 1 (NONE transform)
print("Case 16: Binary mode 1 (NONE transform)")
params16 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [1],
    "class_nodeids": [1],
    "class_treeids": [0],
    "class_weights": [0.8],
    "classlabels_int64s": [0, 1],
    "post_transform": "NONE"
}

x16 = np.random.rand(1000, 1).astype(np.float32)

t0 = time.time()
onnx_labels16, onnx_scores16 = onnx_tree_ensemble_classifier(x16, **params16)
print("ONNX:", onnx_scores16.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op16 = TreeEnsembleClassifierOp(mgr, **params16)
kp_labels16, kp_scores16 = tree_op16.run(x16)
print(f"{tree_op16}:", kp_scores16.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores16.shape == onnx_scores16.shape)
print("Scores Max error:", np.abs(onnx_scores16 - kp_scores16).max())
print("Scores All close:", np.allclose(onnx_scores16, kp_scores16, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels16, kp_labels16))
print("----")

# Case 17: set_attributes
print("Case 17: set_attributes (cover lines 68-87)")
# 先用 params1 初始化，再调用 set_attributes 切换到新参数
params17_new = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_LEQ", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.6, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [3.0, 7.0],
    "classlabels_int64s": [0, 1],
    "base_values": None,
    "post_transform": "SOFTMAX",
}

# 初始用 params1 创建算子
tree_op17 = TreeEnsembleClassifierOp(mgr, **params1)

# 调用 set_attributes 传入所有参数
tree_op17.set_attributes(
    base_values=params17_new["base_values"],
    class_ids=params17_new["class_ids"],
    class_nodeids=params17_new["class_nodeids"],
    class_treeids=params17_new["class_treeids"],
    class_weights=params17_new["class_weights"],
    classlabels_int64s=params17_new["classlabels_int64s"],
    classlabels_strings=None,
    nodes_falsenodeids=params17_new["nodes_falsenodeids"],
    nodes_featureids=params17_new["nodes_featureids"],
    nodes_hitrates=params17_new["nodes_hitrates"],
    nodes_missing_value_tracks_true=params17_new["nodes_missing_value_tracks_true"],
    nodes_modes=params17_new["nodes_modes"],
    nodes_nodeids=params17_new["nodes_nodeids"],
    nodes_treeids=params17_new["nodes_treeids"],
    nodes_truenodeids=params17_new["nodes_truenodeids"],
    nodes_values=params17_new["nodes_values"],
    post_transform=params17_new["post_transform"],
)

x17 = np.random.rand(500, 1).astype(np.float32)

t0 = time.time()
onnx_labels17, onnx_scores17 = onnx_tree_ensemble_classifier(x17, **{k: v for k, v in params17_new.items() if k != "base_values"})
print("ONNX:", onnx_scores17.shape, time.time() - t0, "seconds")

t0 = time.time()
kp_labels17, kp_scores17 = tree_op17.run(x17)
print(f"{tree_op17}:", kp_scores17.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores17.shape == onnx_scores17.shape)
print("Scores Max error:", np.abs(onnx_scores17 - kp_scores17).max())
print("Scores All close:", np.allclose(onnx_scores17, kp_scores17, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels17, kp_labels17))
print("----")

# Case 18: nodes_modes 为 bytes 类型
print("Case 18: nodes_modes as bytes (cover line 121)")
params18 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    # 使用 bytes 类型传入 mode
    "nodes_modes": [b"BRANCH_LT", b"LEAF", b"LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [1.0, 1.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "NONE",
}

# 用于参考的 str 版本
params18_str = dict(params18)
params18_str["nodes_modes"] = ["BRANCH_LT", "LEAF", "LEAF"]

x18 = np.random.rand(500, 1).astype(np.float32)

t0 = time.time()
onnx_labels18, onnx_scores18 = onnx_tree_ensemble_classifier(x18, **params18_str)
print("ONNX:", onnx_scores18.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op18 = TreeEnsembleClassifierOp(mgr, **params18)
kp_labels18, kp_scores18 = tree_op18.run(x18)
print(f"{tree_op18}:", kp_scores18.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores18.shape == onnx_scores18.shape)
print("Scores Max error:", np.abs(onnx_scores18 - kp_scores18).max())
print("Scores All close:", np.allclose(onnx_scores18, kp_scores18, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels18, kp_labels18))
print("----")

# Case 19: 纯 BRANCH_GTE 树
print("Case 19: BRANCH_GTE only tree (cover line 247)")
params19 = {
    "nodes_falsenodeids": [2, 0, 0, 2, 0, 0],
    "nodes_featureids": [0, 0, 0, 0, 0, 0],
    "nodes_hitrates": [1.0] * 6,
    "nodes_missing_value_tracks_true": [0] * 6,
    # 树0: BRANCH_GTE, 树1: BRANCH_NEQ
    "nodes_modes": ["BRANCH_GTE", "LEAF", "LEAF", "BRANCH_GT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2, 0, 1, 2],
    "nodes_treeids": [0, 0, 0, 1, 1, 1],
    "nodes_truenodeids": [1, 0, 0, 1, 0, 0],
    "nodes_falsenodeids": [2, 0, 0, 2, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
    "class_ids": [0, 1, 0, 1],
    "class_nodeids": [1, 2, 1, 2],
    "class_treeids": [0, 0, 1, 1],
    "class_weights": [4.0, 8.0, 2.0, 6.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "NONE",
}

x19 = np.random.rand(500, 1).astype(np.float32)

t0 = time.time()
onnx_labels19, onnx_scores19 = onnx_tree_ensemble_classifier(x19, **params19)
print("ONNX:", onnx_scores19.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op19 = TreeEnsembleClassifierOp(mgr, **params19)
kp_labels19, kp_scores19 = tree_op19.run(x19)
print(f"{tree_op19}:", kp_scores19.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores19.shape == onnx_scores19.shape)
print("Scores Max error:", np.abs(onnx_scores19 - kp_scores19).max())
print("Scores All close:", np.allclose(onnx_scores19, kp_scores19, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels19, kp_labels19))
print("----")

# Case 20: 单独 BRANCH_NEQ 树
print("Case 20: BRANCH_NEQ only tree (cover line 253)")
params20 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_NEQ", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    "class_ids": [0, 1],
    "class_nodeids": [1, 2],
    "class_treeids": [0, 0],
    "class_weights": [10.0, 5.0],
    "classlabels_int64s": [0, 1],
    "post_transform": "NONE",
}

x20 = np.array([[0.5], [0.3], [0.7]], dtype=np.float32)

t0 = time.time()
onnx_labels20, onnx_scores20 = onnx_tree_ensemble_classifier(x20, **params20)
print("ONNX:", onnx_scores20.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op20 = TreeEnsembleClassifierOp(mgr, **params20)
kp_labels20, kp_scores20 = tree_op20.run(x20)
print(f"{tree_op20}:", kp_scores20.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores20.shape == onnx_scores20.shape)
print("Scores Max error:", np.abs(onnx_scores20 - kp_scores20).max())
print("Scores All close:", np.allclose(onnx_scores20, kp_scores20, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels20, kp_labels20))
print("----")

# Case 21: classlabels_int64s=[1] 单元素且值为1
print("Case 21: classlabels_int64s=[1] single-element fuse path (cover line 383)")
params21 = {
    "nodes_falsenodeids": [2, 0, 0],
    "nodes_featureids": [0, 0, 0],
    "nodes_hitrates": [1.0, 1.0, 1.0],
    "nodes_missing_value_tracks_true": [0, 0, 0],
    "nodes_modes": ["BRANCH_LT", "LEAF", "LEAF"],
    "nodes_nodeids": [0, 1, 2],
    "nodes_treeids": [0, 0, 0],
    "nodes_truenodeids": [1, 0, 0],
    "nodes_values": [0.5, 0.0, 0.0],
    # class_ids 只含 [1]（二值模式，unique={1}，num_classes=2）
    "class_ids": [1],
    "class_nodeids": [1],
    "class_treeids": [0],
    "class_weights": [0.9],
    # classlabels_int64s=[1]：len==1 且 [0]==1
    "classlabels_int64s": [1],
    "post_transform": "NONE",
}

x21 = np.random.rand(200, 1).astype(np.float32)

t0 = time.time()
onnx_labels21, onnx_scores21 = onnx_tree_ensemble_classifier(x21, **params21)
print("ONNX:", onnx_scores21.shape, time.time() - t0, "seconds")

t0 = time.time()
tree_op21 = TreeEnsembleClassifierOp(mgr, **params21)
kp_labels21, kp_scores21 = tree_op21.run(x21)
print(f"{tree_op21}:", kp_scores21.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores21.shape == onnx_scores21.shape)
print("Scores Max error:", np.abs(onnx_scores21 - kp_scores21).max())
print("Scores All close:", np.allclose(onnx_scores21, kp_scores21, rtol=1e-4, atol=1e-4))
print("Labels All close:", np.allclose(onnx_labels21, kp_labels21))
print("----")

