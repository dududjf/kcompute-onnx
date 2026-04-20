import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kp
import numpy as np
import time
from kp_onnx_ssbo.kop_tree_ensemble import TreeEnsembleOp
from enum import IntEnum
from typing import Callable, Union, Set

class AggregationFunction(IntEnum):
    AVERAGE = 0
    SUM = 1
    MIN = 2
    MAX = 3

class Mode(IntEnum):
    LEQ = 0
    LT = 1
    GTE = 2
    GT = 3
    EQ = 4
    NEQ = 5
    MEMBER = 6

class Leaf:
    def __init__(self, weight: float, target_id: int) -> None:
        self.weight = weight
        self.target_id = target_id

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.weight, self.target_id])

class Node:
    compare: Callable[[float, Union[float, Set[float]]], bool]
    true_branch: Union['Node', Leaf]
    false_branch: Union['Node', Leaf]
    feature: int

    def __init__(
        self,
        mode: Mode,
        value: Union[float, Set[float]],
        feature: int,
        missing_tracks_true: bool,
    ) -> None:
        if mode == Mode.LEQ:
            self.compare = lambda x: x[feature].item() <= value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.LT:
            self.compare = lambda x: x[feature].item() < value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.GTE:
            self.compare = lambda x: x[feature].item() >= value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.GT:
            self.compare = lambda x: x[feature].item() > value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.EQ:
            self.compare = lambda x: x[feature].item() == value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.NEQ:
            self.compare = lambda x: x[feature].item() != value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        elif mode == Mode.MEMBER:
            self.compare = lambda x: x[feature].item() in value or (
                missing_tracks_true and np.isnan(x[feature].item())
            )
        self.mode = mode
        self.value = value
        self.feature = feature

    def predict(self, x: np.ndarray) -> float:
        if self.compare(x):
            return self.true_branch.predict(x)
        else:
            return self.false_branch.predict(x)

class TreeEnsembleReference:
    def __init__(self, **kwargs):
        self.nodes_splits = kwargs.get("nodes_splits", [])
        self.nodes_featureids = kwargs.get("nodes_featureids", [])
        self.nodes_modes = kwargs.get("nodes_modes", [])
        self.nodes_truenodeids = kwargs.get("nodes_truenodeids", [])
        self.nodes_falsenodeids = kwargs.get("nodes_falsenodeids", [])
        self.nodes_trueleafs = kwargs.get("nodes_trueleafs", [])
        self.nodes_falseleafs = kwargs.get("nodes_falseleafs", [])
        self.leaf_targetids = kwargs.get("leaf_targetids", [])
        self.leaf_weights = kwargs.get("leaf_weights", [])
        self.tree_roots = kwargs.get("tree_roots", [])
        self.n_targets = kwargs.get("n_targets", 1)
        self.aggregate_function = kwargs.get("aggregate_function", "SUM")
        self.post_transform = kwargs.get("post_transform", "NONE")
        assert self.post_transform in [None, "NONE"], f"post_transform must be NONE, got {self.post_transform}"
        self.nodes_missing_value_tracks_true = kwargs.get("nodes_missing_value_tracks_true", [])
        self.membership_values = kwargs.get("membership_values", [])

        self.trees = self._build_trees()

    def _build_trees(self):
        set_membership_iter = (
            iter(self.membership_values) if self.membership_values is not None else None
        )

        def build_node(current_node_index, is_leaf) -> Union[Node, Leaf]:
            if is_leaf:
                return Leaf(
                    self.leaf_weights[current_node_index], self.leaf_targetids[current_node_index]
                )

            mode = self.nodes_modes[current_node_index]
            feature = self.nodes_featureids[current_node_index]
            missing_tracks_true = (
                self.nodes_missing_value_tracks_true[current_node_index]
                if current_node_index < len(self.nodes_missing_value_tracks_true)
                else False
            )

            if mode == Mode.MEMBER:
                set_members = set()
                while True:
                    try:
                        set_member = next(set_membership_iter)
                        if np.isnan(set_member):
                            break
                        set_members.add(set_member)
                    except StopIteration:
                        break
                node = Node(mode, set_members, feature, missing_tracks_true)
            else:
                val = self.nodes_splits[current_node_index]
                node = Node(mode, val, feature, missing_tracks_true)

            node.true_branch = build_node(
                self.nodes_truenodeids[current_node_index],
                self.nodes_trueleafs[current_node_index],
            )
            node.false_branch = build_node(
                self.nodes_falsenodeids[current_node_index],
                self.nodes_falseleafs[current_node_index],
            )
            return node

        trees = []
        for root_index in self.tree_roots:
            is_leaf = False
            if root_index < len(self.nodes_trueleafs):
                 if (self.nodes_trueleafs[root_index] and self.nodes_falseleafs[root_index] and
                     self.nodes_truenodeids[root_index] == self.nodes_falsenodeids[root_index]):
                     is_leaf = True
            trees.append(build_node(root_index, is_leaf))
        return trees

    def predict(self, X):
        raw_values = [
            np.apply_along_axis(tree.predict, axis=1, arr=X) for tree in self.trees
        ]
        weights, target_ids = zip(*[np.split(x, 2, axis=1) for x in raw_values])
        weights = np.concatenate(weights, axis=1)
        target_ids = np.concatenate(target_ids, axis=1).astype(np.int64)

        agg_func_map = {
            "SUM": AggregationFunction.SUM,
            "AVERAGE": AggregationFunction.AVERAGE,
            "MIN": AggregationFunction.MIN,
            "MAX": AggregationFunction.MAX
        }
        agg_func = agg_func_map.get(self.aggregate_function, AggregationFunction.SUM)

        if agg_func in (AggregationFunction.SUM, AggregationFunction.AVERAGE):
            result = np.zeros((len(X), self.n_targets), dtype=X.dtype)
        elif agg_func == AggregationFunction.MIN:
            result = np.full((len(X), self.n_targets), np.finfo(X.dtype).max)
        elif agg_func == AggregationFunction.MAX:
            result = np.full((len(X), self.n_targets), np.finfo(X.dtype).min)

        for batch_num, (w, t) in enumerate(zip(weights, target_ids)):
            weight = w.reshape(-1)
            target_id = t.reshape(-1)
            if agg_func == AggregationFunction.SUM:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] += value
            elif agg_func == AggregationFunction.AVERAGE:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] += value / len(self.trees)
            elif agg_func == AggregationFunction.MIN:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] = min(result[batch_num, tid], value)
            elif agg_func == AggregationFunction.MAX:
                for value, tid in zip(weight, target_id):
                    result[batch_num, tid] = max(result[batch_num, tid], value)

        return result

mgr = kp.Manager()

def gen_data(rows, cols):
    return np.random.rand(rows, cols).astype(np.float32)

# Case 1: Simple Tree (SUM)
print("Case 1: Simple Tree (SUM)")
params1 = {
    "nodes_splits": [0.5],
    "nodes_featureids": [0],
    "nodes_modes": [0], # LEQ
    "nodes_truenodeids": [0],
    "nodes_falsenodeids": [1],
    "nodes_trueleafs": [1],
    "nodes_falseleafs": [1],
    "leaf_targetids": [0, 0],
    "leaf_weights": [10.0, 20.0],
    "tree_roots": [0],
    "n_targets": 1,
    "aggregate_function": "SUM"
}

x1 = gen_data(1000, 1)

t0 = time.time()
ref1 = TreeEnsembleReference(**params1)
onnx_scores1 = ref1.predict(x1)
print("ONNX:", onnx_scores1.shape, time.time() - t0, "seconds")

t0 = time.time()
op1 = TreeEnsembleOp(mgr, **params1)
kp_scores1 = op1.run(x1)[0]
print(f"{op1}:", kp_scores1.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores1.shape == onnx_scores1.shape)
print("Scores Max error:", np.abs(onnx_scores1 - kp_scores1).max())
print("Scores All close:", np.allclose(onnx_scores1, kp_scores1, rtol=1e-4, atol=1e-4))
print("----")

# Case 2: Two Trees, Multi-target (AVERAGE)
print("Case 2: Two Trees, Multi-target (AVERAGE)")
params2 = {
    "nodes_splits": [0.5, 0.5],
    "nodes_featureids": [0, 0],
    "nodes_modes": [0, 0], # LEQ
    "nodes_truenodeids": [0, 2],
    "nodes_falsenodeids": [1, 3],
    "nodes_trueleafs": [1, 1],
    "nodes_falseleafs": [1, 1],
    "leaf_targetids": [0, 1, 0, 1],
    "leaf_weights": [10.0, 20.0, 30.0, 40.0],
    "tree_roots": [0, 1],
    "n_targets": 2,
    "aggregate_function": "AVERAGE"
}

x2 = gen_data(1000, 1)

t0 = time.time()
ref2 = TreeEnsembleReference(**params2)
onnx_scores2 = ref2.predict(x2)
print("ONNX:", onnx_scores2.shape, time.time() - t0, "seconds")

t0 = time.time()
op2 = TreeEnsembleOp(mgr, **params2)
kp_scores2 = op2.run(x2)[0]
print(f"{op2}:", kp_scores2.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores2.shape == onnx_scores2.shape)
print("Scores Max error:", np.abs(onnx_scores2 - kp_scores2).max())
print("Scores All close:", np.allclose(onnx_scores2, kp_scores2, rtol=1e-4, atol=1e-4))
print("----")

# Case 3: MIN Aggregation
print("Case 3: MIN Aggregation")
params3 = dict(params2)
params3["aggregate_function"] = "MIN"

x3 = gen_data(1000, 1)

t0 = time.time()
ref3 = TreeEnsembleReference(**params3)
onnx_scores3 = ref3.predict(x3)
print("ONNX:", onnx_scores3.shape, time.time() - t0, "seconds")

t0 = time.time()
op3 = TreeEnsembleOp(mgr, **params3)
kp_scores3 = op3.run(x3)[0]
print(f"{op3}:", kp_scores3.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores3.shape == onnx_scores3.shape)
print("Scores Max error:", np.abs(onnx_scores3 - kp_scores3).max())
print("Scores All close:", np.allclose(onnx_scores3, kp_scores3, rtol=1e-4, atol=1e-4))
print("----")

# Case 4: MAX Aggregation
print("Case 4: MAX Aggregation")
params4 = dict(params2)
params4["aggregate_function"] = "MAX"

x4 = gen_data(1000, 1)

t0 = time.time()
ref4 = TreeEnsembleReference(**params4)
onnx_scores4 = ref4.predict(x4)
print("ONNX:", onnx_scores4.shape, time.time() - t0, "seconds")

t0 = time.time()
op4 = TreeEnsembleOp(mgr, **params4)
kp_scores4 = op4.run(x4)[0]
print(f"{op4}:", kp_scores4.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores4.shape == onnx_scores4.shape)
print("Scores Max error:", np.abs(onnx_scores4 - kp_scores4).max())
print("Scores All close:", np.allclose(onnx_scores4, kp_scores4, rtol=1e-4, atol=1e-4))
print("----")

# Case 5: Missing Values
print("Case 5: Missing Values")
params5 = dict(params1)
params5["nodes_missing_value_tracks_true"] = [1] # NaN -> True

x5 = gen_data(1000, 1)
x5[::2, 0] = np.nan

t0 = time.time()
ref5 = TreeEnsembleReference(**params5)
onnx_scores5 = ref5.predict(x5)
print("ONNX:", onnx_scores5.shape, time.time() - t0, "seconds")

t0 = time.time()
op5 = TreeEnsembleOp(mgr, **params5)
kp_scores5 = op5.run(x5)[0]
print(f"{op5}:", kp_scores5.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores5.shape == onnx_scores5.shape)
print("Scores Max error:", np.abs(onnx_scores5 - kp_scores5).max())
print("Scores All close:", np.allclose(onnx_scores5, kp_scores5, rtol=1e-4, atol=1e-4))
print("----")

# Case 6: Member Mode
print("Case 6: Member Mode")
# Tree 0:
# Root (0): Feature 0 in {1.0, 2.0, 3.0}
#   True -> Leaf 0 (Target 0, Weight 10.0)
#   False -> Leaf 1 (Target 0, Weight 20.0)
params6 = {
    "nodes_splits": [0.0], # Ignored for MEMBER
    "nodes_featureids": [0],
    "nodes_modes": [6], # MEMBER
    "nodes_truenodeids": [0],
    "nodes_falsenodeids": [1],
    "nodes_trueleafs": [1],
    "nodes_falseleafs": [1],
    "leaf_targetids": [0, 0],
    "leaf_weights": [10.0, 20.0],
    "tree_roots": [0],
    "n_targets": 1,
    "aggregate_function": "SUM",
    "membership_values": [1.0, 2.0, 3.0, np.nan]
}

x6 = np.array([[1.0], [2.0], [3.0], [4.0], [0.0]], dtype=np.float32)

t0 = time.time()
ref6 = TreeEnsembleReference(**params6)
onnx_scores6 = ref6.predict(x6)
print("ONNX:", onnx_scores6.shape, time.time() - t0, "seconds")

t0 = time.time()
op6 = TreeEnsembleOp(mgr, **params6)
kp_scores6 = op6.run(x6)[0]
print(f"{op6}:", kp_scores6.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores6.shape == onnx_scores6.shape)
print("Scores Max error:", np.abs(onnx_scores6 - kp_scores6).max())
print("Scores All close:", np.allclose(onnx_scores6, kp_scores6, rtol=1e-4, atol=1e-4))
print("----")

# Case 7: All Comparison Modes
print("Case 7: All Comparison Modes")
# Tree 0: LT (1) 0.5
# Tree 1: GTE (2) 0.5
# Tree 2: GT (3) 0.5
# Tree 3: EQ (4) 0.5
# Tree 4: NEQ (5) 0.5
params7 = {
    "nodes_splits": [0.5, 0.5, 0.5, 0.5, 0.5],
    "nodes_featureids": [0, 0, 0, 0, 0],
    "nodes_modes": [1, 2, 3, 4, 5],
    "nodes_truenodeids": [0, 1, 2, 3, 4], # Leaf indices
    "nodes_falsenodeids": [5, 6, 7, 8, 9], # Leaf indices
    "nodes_trueleafs": [1, 1, 1, 1, 1],
    "nodes_falseleafs": [1, 1, 1, 1, 1],
    "leaf_targetids": [0] * 10,
    "leaf_weights": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "tree_roots": [0, 1, 2, 3, 4],
    "n_targets": 1,
    "aggregate_function": "SUM"
}

x7 = np.array([[0.4], [0.5], [0.6]], dtype=np.float32)
# Expected:
# 0.4: LT(T), GTE(F), GT(F), EQ(F), NEQ(T) -> 1+0+0+0+1 = 2.0
# 0.5: LT(F), GTE(T), GT(F), EQ(T), NEQ(F) -> 0+1+0+1+0 = 2.0
# 0.6: LT(F), GTE(T), GT(T), EQ(F), NEQ(T) -> 0+1+1+0+1 = 3.0

t0 = time.time()
ref7 = TreeEnsembleReference(**params7)
onnx_scores7 = ref7.predict(x7)
print("ONNX:", onnx_scores7.shape, time.time() - t0, "seconds")

t0 = time.time()
op7 = TreeEnsembleOp(mgr, **params7)
kp_scores7 = op7.run(x7)[0]
print(f"{op7}:", kp_scores7.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores7.shape == onnx_scores7.shape)
print("Scores Max error:", np.abs(onnx_scores7 - kp_scores7).max())
print("Scores All close:", np.allclose(onnx_scores7, kp_scores7, rtol=1e-4, atol=1e-4))
print("----")

# Case 8: Missing Tracks False
print("Case 8: Missing Tracks False")
params8 = {
    "nodes_splits": [0.5],
    "nodes_featureids": [0],
    "nodes_modes": [0], # LEQ
    "nodes_missing_value_tracks_true": [0], # False
    "nodes_truenodeids": [0],
    "nodes_falsenodeids": [1],
    "nodes_trueleafs": [1],
    "nodes_falseleafs": [1],
    "leaf_targetids": [0, 0],
    "leaf_weights": [10.0, 20.0],
    "tree_roots": [0],
    "n_targets": 1,
    "aggregate_function": "SUM"
}

x8 = np.array([[np.nan]], dtype=np.float32)

t0 = time.time()
ref8 = TreeEnsembleReference(**params8)
onnx_scores8 = ref8.predict(x8)
print("ONNX:", onnx_scores8.shape, time.time() - t0, "seconds")

t0 = time.time()
op8 = TreeEnsembleOp(mgr, **params8)
kp_scores8 = op8.run(x8)[0]
print(f"{op8}:", kp_scores8.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores8.shape == onnx_scores8.shape)
print("Scores Max error:", np.abs(onnx_scores8 - kp_scores8).max())
print("Scores All close:", np.allclose(onnx_scores8, kp_scores8, rtol=1e-4, atol=1e-4))
print("----")

# Case 9: Root is Leaf
print("Case 9: Root is Leaf")
params9 = {
    "nodes_splits": [0.0],
    "nodes_featureids": [0],
    "nodes_modes": [0],
    "nodes_truenodeids": [0],
    "nodes_falsenodeids": [0],
    "nodes_trueleafs": [1],
    "nodes_falseleafs": [1],
    "leaf_targetids": [0],
    "leaf_weights": [42.0],
    "tree_roots": [0],
    "n_targets": 1,
    "aggregate_function": "SUM"
}

x9 = gen_data(10, 1)

t0 = time.time()
ref9 = TreeEnsembleReference(**params9)
onnx_scores9 = ref9.predict(x9)
print("ONNX:", onnx_scores9.shape, time.time() - t0, "seconds")

t0 = time.time()
op9 = TreeEnsembleOp(mgr, **params9)
kp_scores9 = op9.run(x9)[0]
print(f"{op9}:", kp_scores9.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores9.shape == onnx_scores9.shape)
print("Scores Max error:", np.abs(onnx_scores9 - kp_scores9).max())
print("Scores All close:", np.allclose(onnx_scores9, kp_scores9, rtol=1e-4, atol=1e-4))
print("----")

# Case 10: Member Mode Empty
print("Case 10: Member Mode Empty")
params10 = {
    "nodes_splits": [0.0],
    "nodes_featureids": [0],
    "nodes_modes": [6], # MEMBER
    "nodes_truenodeids": [0],
    "nodes_falsenodeids": [1],
    "nodes_trueleafs": [1],
    "nodes_falseleafs": [1],
    "leaf_targetids": [0, 0],
    "leaf_weights": [10.0, 20.0],
    "tree_roots": [0],
    "n_targets": 1,
    "aggregate_function": "SUM",
    "membership_values": []
}

x10 = np.array([[1.0]], dtype=np.float32)

t0 = time.time()
ref10 = TreeEnsembleReference(**params10)
onnx_scores10 = ref10.predict(x10)
print("ONNX:", onnx_scores10.shape, time.time() - t0, "seconds")

t0 = time.time()
op10 = TreeEnsembleOp(mgr, **params10)
kp_scores10 = op10.run(x10)[0]
print(f"{op10}:", kp_scores10.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores10.shape == onnx_scores10.shape)
print("Scores Max error:", np.abs(onnx_scores10 - kp_scores10).max())
print("Scores All close:", np.allclose(onnx_scores10, kp_scores10, rtol=1e-4, atol=1e-4))
print("----")

# Case 11: set_attributes — 覆盖 61-79 行
# 先用 params1 初始化，再调用 set_attributes 切换到新参数（2棵树、MAX 聚合）
print("Case 11: set_attributes (switch from params1 to new MAX params)")

params11_new = {
    "nodes_splits": [0.3, 0.7],
    "nodes_featureids": [0, 0],
    "nodes_modes": [0, 0],  # LEQ
    "nodes_truenodeids": [0, 2],
    "nodes_falsenodeids": [1, 3],
    "nodes_trueleafs": [1, 1],
    "nodes_falseleafs": [1, 1],
    "leaf_targetids": [0, 0, 0, 0],
    "leaf_weights": [5.0, 15.0, 10.0, 25.0],
    "tree_roots": [0, 1],
    "n_targets": 1,
    "aggregate_function": "MAX",
    "nodes_missing_value_tracks_true": [0, 0],
    "membership_values": None,
}

x11 = gen_data(500, 1)

# 初始用 params1 创建算子
op11 = TreeEnsembleOp(mgr, **params1)

# 通过 set_attributes 切换到新参数（覆盖 61-79 行所有赋值分支）
op11.set_attributes(
    nodes_splits=params11_new["nodes_splits"],
    nodes_featureids=params11_new["nodes_featureids"],
    nodes_modes=params11_new["nodes_modes"],
    nodes_truenodeids=params11_new["nodes_truenodeids"],
    nodes_falsenodeids=params11_new["nodes_falsenodeids"],
    nodes_trueleafs=params11_new["nodes_trueleafs"],
    nodes_falseleafs=params11_new["nodes_falseleafs"],
    leaf_targetids=params11_new["leaf_targetids"],
    leaf_weights=params11_new["leaf_weights"],
    tree_roots=params11_new["tree_roots"],
    n_targets=params11_new["n_targets"],
    aggregate_function=params11_new["aggregate_function"],
    post_transform="NONE",
    nodes_missing_value_tracks_true=params11_new["nodes_missing_value_tracks_true"],
    membership_values=params11_new["membership_values"],
)

t0 = time.time()
ref11 = TreeEnsembleReference(**params11_new)
onnx_scores11 = ref11.predict(x11)
print("ONNX:", onnx_scores11.shape, time.time() - t0, "seconds")

t0 = time.time()
kp_scores11 = op11.run(x11)[0]
print(f"{op11}:", kp_scores11.shape, time.time() - t0, "seconds")

print("Scores shape equal:", kp_scores11.shape == onnx_scores11.shape)
print("Scores Max error:", np.abs(onnx_scores11 - kp_scores11).max())
print("Scores All close:", np.allclose(onnx_scores11, kp_scores11, rtol=1e-4, atol=1e-4))
print("----")

