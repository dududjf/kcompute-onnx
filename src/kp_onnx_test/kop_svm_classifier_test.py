from kp import Manager
import numpy as np
import time
from kp_onnx_ssbo.kop_svm_classifier import SVMClassifierOp


# --------------------------
# ONNX SVM Classifier baseline (copied from op_svm_classifier.py and op_svm_helper.py)
# --------------------------
class SVMAttributes:
    def __init__(self):
        self._names = []

    def add(self, name, value):
        if isinstance(value, list) and name not in {"kernel_params"}:
            if name in {"vectors_per_class"}:
                value = np.array(value, dtype=np.int64)
            else:
                value = np.array(value, dtype=np.float32)
        setattr(self, name, value)


class SVMCommon:
    def __init__(self, **kwargs):
        self.atts = SVMAttributes()
        for name, value in kwargs.items():
            self.atts.add(name, value)

        if getattr(self.atts, "kernel_params", None) is not None:
            self.gamma_ = self.atts.kernel_params[0]
            self.coef0_ = self.atts.kernel_params[1]
            self.degree_ = int(self.atts.kernel_params[2])
        else:
            self.gamma_ = 0.0
            self.coef0_ = 0.0
            self.degree_ = 0

    def kernel_dot(self, pA, pB, kernel):
        k = kernel.lower()
        if k == "poly":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return s ** self.degree_
        if k == "sigmoid":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return np.tanh(s)
        if k == "rbf":
            diff = pA - pB
            s = (diff * diff).sum()
            return np.exp(-self.gamma_ * s)
        if k == "linear":
            return np.dot(pA, pB)
        raise ValueError(f"Unexpected kernel={kernel!r}.")


def onnx_reference_svm_classifier_linear(X, coefs, class_count_, kernel_type_, rho):
    """LINEAR mode: each class has a weight vector"""
    svm = SVMCommon(kernel_type=kernel_type_, rho=rho)

    scores_list = []
    for j in range(class_count_):
        d = svm.kernel_dot(X, coefs[j], kernel_type_)
        score = rho[0] + d
        scores_list.append(score)
    return np.array(scores_list, dtype=X.dtype)


def onnx_reference_svm_classifier_svc(
    X, sv, vector_count_, kernel_type_, class_count_, starting_vector_, coefs, rho, kernel_params
):
    """SVC mode: one-vs-one classification with support vectors"""
    svm = SVMCommon(kernel_type=kernel_type_, kernel_params=kernel_params, rho=rho)

    evals = 0
    kernels_list = [svm.kernel_dot(X, sv[j], kernel_type_) for j in range(vector_count_)]
    kernels = np.array(kernels_list)

    votes = np.zeros((class_count_,), dtype=X.dtype)
    scores = []

    for i in range(class_count_):
        si_i = starting_vector_[i]
        class_i_sc = len([k for k, v in enumerate(starting_vector_) if v == si_i])
        if i + 1 < len(starting_vector_):
            class_i_sc = starting_vector_[i + 1] - si_i
        else:
            class_i_sc = vector_count_ - si_i

        for j in range(i + 1, class_count_):
            si_j = starting_vector_[j]
            if j + 1 < len(starting_vector_):
                class_j_sc = starting_vector_[j + 1] - si_j
            else:
                class_j_sc = vector_count_ - si_j

            s1 = np.dot(
                coefs[j - 1, si_i:si_i + class_i_sc],
                kernels[si_i:si_i + class_i_sc],
            )
            s2 = np.dot(
                coefs[i, si_j:si_j + class_j_sc],
                kernels[si_j:si_j + class_j_sc],
            )

            s = rho[evals] + s1 + s2
            scores.append(s)
            if s > 0:
                votes[i] += 1
            else:
                votes[j] += 1
            evals += 1

    return votes, np.array(scores, dtype=X.dtype)


def sigmoid_probability(score, proba, probb):
    """Apply sigmoid probability transformation to a score."""
    val = score * proba + probb
    abs_val = np.abs(val)
    logistic_val = 1.0 / (1.0 + np.exp(-abs_val))
    logistic_val = (1.0 - logistic_val) if val < 0 else logistic_val
    return 1.0 - logistic_val


def onnx_reference_svm_classifier(
    X,
    classlabels_ints=None,
    coefficients=None,
    kernel_params=None,
    kernel_type=None,
    rho=None,
    support_vectors=None,
    vectors_per_class=None,
    prob_a=None,
    prob_b=None,
    post_transform="NONE",
):
    X = np.array(X).astype(np.float32)
    X_flat = X.reshape((X.shape[0], -1))

    class_count_ = max(len(classlabels_ints or []), 1)
    vector_count_ = 0
    starting_vector_ = []

    if vectors_per_class is not None:
        for vc in vectors_per_class:
            starting_vector_.append(vector_count_)
            vector_count_ += vc

    has_proba = prob_a is not None and prob_b is not None

    if vector_count_ > 0:
        # SVC mode
        sv = np.array(support_vectors, dtype=np.float32).reshape((vector_count_, -1))
        kernel_type_ = kernel_type
        coefs = np.array(coefficients, dtype=np.float32).reshape((-1, vector_count_))

        res_scores = np.empty((X_flat.shape[0], class_count_ * (class_count_ - 1) // 2), dtype=X.dtype)
        res_votes = np.empty((X_flat.shape[0], class_count_), dtype=X.dtype)

        for n in range(X_flat.shape[0]):
            vote, scores = onnx_reference_svm_classifier_svc(
                X_flat[n], sv, vector_count_, kernel_type_, class_count_,
                starting_vector_, coefs, rho, kernel_params
            )
            res_scores[n, :] = scores
            res_votes[n, :] = vote

        # Apply probability transformation if has_proba
        if has_proba:
            num_pairs = class_count_ * (class_count_ - 1) // 2
            for n in range(X_flat.shape[0]):
                for pair_idx in range(num_pairs):
                    res_scores[n, pair_idx] = sigmoid_probability(
                        res_scores[n, pair_idx], prob_a[pair_idx], prob_b[pair_idx]
                    )

            # Recompute votes from probability-transformed scores
            res_votes[:] = 0
            for n in range(X_flat.shape[0]):
                pair_idx = 0
                for i in range(class_count_):
                    for j in range(i + 1, class_count_):
                        if res_scores[n, pair_idx] > 0:
                            res_votes[n, i] += 1
                        else:
                            res_votes[n, j] += 1
                        pair_idx += 1

        # Apply post_transform to votes if needed (for label computation)
        votes_for_labels = res_votes.copy()
        if post_transform.upper() == "SOFTMAX":
            for i in range(votes_for_labels.shape[0]):
                v_max = votes_for_labels[i].max()
                votes_for_labels[i] = np.exp(votes_for_labels[i] - v_max)
                votes_for_labels[i] /= votes_for_labels[i].sum()
        elif post_transform.upper() == "LOGISTIC":
            for i in range(votes_for_labels.shape[0]):
                for j in range(class_count_):
                    val = votes_for_labels[i, j]
                    abs_val = np.abs(val)
                    v = 1.0 / (1.0 + np.exp(-abs_val))
                    votes_for_labels[i, j] = (1.0 - v) if val < 0 else v
        elif post_transform.upper() == "SOFTMAX_ZERO":
            for i in range(votes_for_labels.shape[0]):
                v_max = votes_for_labels[i].max()
                exp_neg_v_max = np.exp(-v_max)
                for j in range(class_count_):
                    v = votes_for_labels[i, j]
                    if v > 0.0000001 or v < -0.0000001:
                        votes_for_labels[i, j] = np.exp(v - v_max)
                    else:
                        votes_for_labels[i, j] = v * exp_neg_v_max
                s = votes_for_labels[i].sum()
                votes_for_labels[i] /= s if s != 0 else 1.0
        elif post_transform.upper() == "PROBIT":
            for i in range(votes_for_labels.shape[0]):
                for j in range(class_count_):
                    val = votes_for_labels[i, j]
                    x = val * 2 - 1
                    sgn = -1.0 if x < 0 else 1.0
                    x_abs = (1.0 - x) * (1 + x)
                    if x_abs == 0:
                        votes_for_labels[i, j] = 0
                    else:
                        log_val = np.log(x_abs)
                        v = 2.0 / (np.pi * 0.147) + 0.5 * log_val
                        v2 = (1.0 / 0.147) * log_val
                        v3 = -v + np.sqrt(v * v - v2)
                        votes_for_labels[i, j] = 1.41421356 * sgn * np.sqrt(v3)

        # Compute labels from transformed votes
        labels = np.argmax(votes_for_labels, axis=1)
        if classlabels_ints is not None and len(classlabels_ints) > 0:
            labels = np.array([classlabels_ints[i] for i in labels], dtype=np.int64)

        return labels, res_scores
    else:
        # SVM_LINEAR mode: vector_count == 0
        # Formula: for each class j, score[j] = X · coefficients[j] + rho[0]
        # Output: (n_samples, class_count) scores

        kernel_type_ = "LINEAR"
        coefs = np.array(coefficients, dtype=np.float32).reshape((class_count_, -1))
        rho0 = rho[0] if rho is not None and len(rho) > 0 else 0.0

        res_scores = np.empty((X_flat.shape[0], class_count_), dtype=X.dtype)
        for n in range(X_flat.shape[0]):
            scores = onnx_reference_svm_classifier_linear(
                X_flat[n], coefs, class_count_, kernel_type_, [rho0]
            )
            res_scores[n, :] = scores

        # Apply post_transform to scores if needed
        if post_transform.upper() == "SOFTMAX":
            for i in range(res_scores.shape[0]):
                v_max = res_scores[i].max()
                res_scores[i] = np.exp(res_scores[i] - v_max)
                res_scores[i] /= res_scores[i].sum()
        elif post_transform.upper() == "LOGISTIC":
            for i in range(res_scores.shape[0]):
                for j in range(class_count_):
                    val = res_scores[i, j]
                    abs_val = np.abs(val)
                    v = 1.0 / (1.0 + np.exp(-abs_val))
                    res_scores[i, j] = (1.0 - v) if val < 0 else v
        elif post_transform.upper() == "SOFTMAX_ZERO":
            for i in range(res_scores.shape[0]):
                v_max = res_scores[i].max()
                exp_neg_v_max = np.exp(-v_max)
                for j in range(class_count_):
                    v = res_scores[i, j]
                    if v > 0.0000001 or v < -0.0000001:
                        res_scores[i, j] = np.exp(v - v_max)
                    else:
                        res_scores[i, j] = v * exp_neg_v_max
                s = res_scores[i].sum()
                res_scores[i] /= s if s != 0 else 1.0
        elif post_transform.upper() == "PROBIT":
            for i in range(res_scores.shape[0]):
                for j in range(class_count_):
                    val = res_scores[i, j]
                    x = val * 2 - 1
                    sgn = -1.0 if x < 0 else 1.0
                    x_abs = (1.0 - x) * (1 + x)
                    if x_abs == 0:
                        res_scores[i, j] = 0
                    else:
                        log_val = np.log(x_abs)
                        v = 2.0 / (np.pi * 0.147) + 0.5 * log_val
                        v2 = (1.0 / 0.147) * log_val
                        v3 = -v + np.sqrt(v * v - v2)
                        res_scores[i, j] = 1.41421356 * sgn * np.sqrt(v3)

        # Compute labels from scores (argmax)
        labels = np.argmax(res_scores, axis=1)
        if classlabels_ints is not None and len(classlabels_ints) > 0:
            labels = np.array([classlabels_ints[i] for i in labels], dtype=np.int64)

        return labels, res_scores


device_id = 0
mgr = Manager(device_id)
print(mgr.get_device_properties())

svm_op = SVMClassifierOp(mgr)

# Case 1: SVM_LINEAR mode (no support vectors) — formula: X·coefficients[j]+rho[0]
print("Case 1: SVM_LINEAR mode (no support vectors) — formula: X·coefficients[j]+rho[0]")
numpy_in = np.random.uniform(-1, 1, (100, 64)).astype(np.float32)
class_count = 4
# In LINEAR mode, coefficients shape is (class_count, feature_size)
coefficients = np.random.uniform(-1, 1, (class_count, 64)).astype(np.float32).reshape(-1)
rho = [3.14159]  # Only use rho[0] in LINEAR mode
classlabels_ints = [0, 1, 2, 3]

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=None,
    kernel_type="LINEAR",  # LINEAR is always used when vector_count=0
    rho=rho,
    support_vectors=None,
    vectors_per_class=None  # ← key: no support vectors, triggers SVM_LINEAR mode
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")
print("NumPy scores (first sample):", numpy_scores[0])
print("NumPy unique labels:", np.unique(numpy_labels))

start_time = time.time()
svm_op.classlabels_ints = classlabels_ints
svm_op.classlabels_strings = None
svm_op.coefficients = coefficients
svm_op.kernel_params = None
svm_op.kernel_type = "LINEAR"
svm_op.post_transform = "NONE"
svm_op.rho = rho
svm_op.support_vectors = None
svm_op.vectors_per_class = None
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Kompute scores (first sample):", kp_scores[0])
print("Kompute unique labels:", np.unique(kp_labels))

print("Labels match:", np.array_equal(numpy_labels, kp_labels))
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 2: SVC mode with LINEAR kernel
print("Case 2: SVC mode with LINEAR kernel")
numpy_in = np.random.uniform(-1, 1, (1024, 256)).astype(np.float32)
class_count = 3
vectors_per_class = [32, 32, 32]
vector_count = sum(vectors_per_class)
support_vectors = np.random.uniform(-1, 1, (vector_count, 256)).astype(np.float32).reshape(-1)
coefficients = np.random.uniform(-1, 1, ((class_count - 1), vector_count)).astype(np.float32).reshape(-1)
rho = [0.1, 0.2, 0.3]  # 3 pairs for 3 classes: (0,1), (0,2), (1,2)
classlabels_ints = [0, 1, 2]

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=None,
    kernel_type="LINEAR",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.classlabels_ints = classlabels_ints
svm_op.coefficients = coefficients
svm_op.kernel_params = None
svm_op.kernel_type = "LINEAR"
svm_op.post_transform = "NONE"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = rho
svm_op.support_vectors = support_vectors
svm_op.vectors_per_class = vectors_per_class
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels match:", np.array_equal(numpy_labels, kp_labels))
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 3: SVC mode with POLY kernel (degree=2)
print("Case 3: SVC mode with POLY kernel (degree=2)")
numpy_in = np.random.uniform(-1, 1, (1024, 128)).astype(np.float32)
class_count = 3
vectors_per_class = [24, 24, 24]
vector_count = sum(vectors_per_class)
support_vectors = np.random.uniform(-1, 1, (vector_count, 128)).astype(np.float32).reshape(-1)
coefficients = np.random.uniform(-1, 1, ((class_count - 1), vector_count)).astype(np.float32).reshape(-1)
kernel_params = [0.3, 0.5, 2]
rho = [0.0, 0.1, -0.1]
classlabels_ints = [0, 1, 2]

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="POLY",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.classlabels_ints = classlabels_ints
svm_op.coefficients = coefficients
svm_op.kernel_params = kernel_params
svm_op.kernel_type = "POLY"
svm_op.post_transform = "NONE"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = rho
svm_op.support_vectors = support_vectors
svm_op.vectors_per_class = vectors_per_class
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels match:", np.array_equal(numpy_labels, kp_labels))
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 4: SVC mode with RBF kernel
print("Case 4: SVC mode with RBF kernel")
numpy_in = np.random.uniform(-1, 1, (1024, 128)).astype(np.float32)
class_count = 3
vectors_per_class = [24, 24, 24]
vector_count = sum(vectors_per_class)
support_vectors = np.random.uniform(-1, 1, (vector_count, 128)).astype(np.float32).reshape(-1)
coefficients = np.random.uniform(-1, 1, ((class_count - 1), vector_count)).astype(np.float32).reshape(-1)
kernel_params = [0.5, 0.0, 0]
rho = [0.1, -0.05, 0.15]
classlabels_ints = [0, 1, 2]

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.classlabels_ints = classlabels_ints
svm_op.coefficients = coefficients
svm_op.kernel_params = kernel_params
svm_op.kernel_type = "RBF"
svm_op.post_transform = "NONE"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = rho
svm_op.support_vectors = support_vectors
svm_op.vectors_per_class = vectors_per_class
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels match:", np.array_equal(numpy_labels, kp_labels))
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 5: SVC mode with SIGMOID kernel
print("Case 5: SVC mode with SIGMOID kernel")
numpy_in = np.random.uniform(-1, 1, (1024, 128)).astype(np.float32)
class_count = 3
vectors_per_class = [24, 24, 24]
vector_count = sum(vectors_per_class)
support_vectors = np.random.uniform(-1, 1, (vector_count, 128)).astype(np.float32).reshape(-1)
coefficients = np.random.uniform(-1, 1, ((class_count - 1), vector_count)).astype(np.float32).reshape(-1)
kernel_params = [0.4, -0.3, 0]
rho = [0.08, -0.02, 0.12]
classlabels_ints = [0, 1, 2]

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="SIGMOID",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.classlabels_ints = classlabels_ints
svm_op.coefficients = coefficients
svm_op.kernel_params = kernel_params
svm_op.kernel_type = "SIGMOID"
svm_op.post_transform = "NONE"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = rho
svm_op.support_vectors = support_vectors
svm_op.vectors_per_class = vectors_per_class
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels match:", np.array_equal(numpy_labels, kp_labels))
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 6: SVC mode with RBF kernel and SOFTMAX post_transform
print("Case 6: SVC mode with RBF kernel and SOFTMAX post_transform")
numpy_in = np.random.uniform(-1, 1, (512, 256)).astype(np.float32)
class_count = 3
vectors_per_class = [32, 32, 32]
vector_count = sum(vectors_per_class)
support_vectors = np.random.uniform(-1, 1, (vector_count, 256)).astype(np.float32).reshape(-1)
coefficients = np.random.uniform(-1, 1, ((class_count - 1), vector_count)).astype(np.float32).reshape(-1)
kernel_params = [0.001, 0.0, 3]
rho = [0.1, -0.2, 0.3]
classlabels_ints = [0, 1, 2]

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.classlabels_ints = classlabels_ints
svm_op.coefficients = coefficients
svm_op.kernel_params = kernel_params
svm_op.kernel_type = "RBF"
svm_op.post_transform = "SOFTMAX"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = rho
svm_op.support_vectors = support_vectors
svm_op.vectors_per_class = vectors_per_class
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels shape match:", kp_labels.shape == numpy_labels.shape)
print("Scores shape match:", kp_scores.shape == numpy_scores.shape)
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 7: SVC mode with RBF kernel and LOGISTIC post_transform
print("Case 7: SVC mode with RBF kernel and LOGISTIC post_transform")
numpy_in = np.random.uniform(-1, 1, (512, 256)).astype(np.float32)

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.post_transform = "LOGISTIC"
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels shape match:", kp_labels.shape == numpy_labels.shape)
print("Scores shape match:", kp_scores.shape == numpy_scores.shape)
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 8: SVC mode with RBF kernel and SOFTMAX_ZERO post_transform
print("Case 8: SVC mode with RBF kernel and SOFTMAX_ZERO post_transform")
numpy_in = np.random.uniform(-1, 1, (512, 256)).astype(np.float32)

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.post_transform = "SOFTMAX_ZERO"
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels shape match:", kp_labels.shape == numpy_labels.shape)
print("Scores shape match:", kp_scores.shape == numpy_scores.shape)
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 9: SVC mode with RBF kernel and PROBIT post_transform
print("Case 9: SVC mode with RBF kernel and PROBIT post_transform")
numpy_in = np.random.uniform(-1, 1, (512, 256)).astype(np.float32)

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.post_transform = "PROBIT"
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels shape match:", kp_labels.shape == numpy_labels.shape)
print("Scores shape match:", kp_scores.shape == numpy_scores.shape)
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 10: SVC mode with RBF kernel and probability coefficients
print("Case 10: SVC mode with RBF kernel and probability coefficients")
numpy_in = np.random.uniform(-1, 1, (512, 256)).astype(np.float32)
num_pairs = 3
prob_a = np.random.uniform(0.1, 0.5, num_pairs).astype(np.float32)
prob_b = np.random.uniform(-0.5, 0.5, num_pairs).astype(np.float32)

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class,
    prob_a=prob_a,
    prob_b=prob_b,
    post_transform="NONE"
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.post_transform = "NONE"
svm_op.prob_a = prob_a
svm_op.prob_b = prob_b
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels shape match:", kp_labels.shape == numpy_labels.shape)
print("Scores shape match:", kp_scores.shape == numpy_scores.shape)
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 11: SVC mode with probability coefficients and SOFTMAX post_transform
print("Case 11: SVC mode with probability coefficients and SOFTMAX post_transform")
numpy_in = np.random.uniform(-1, 1, (512, 256)).astype(np.float32)

start_time = time.time()
numpy_labels, numpy_scores = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=classlabels_ints,
    coefficients=coefficients,
    kernel_params=kernel_params,
    kernel_type="RBF",
    rho=rho,
    support_vectors=support_vectors,
    vectors_per_class=vectors_per_class,
    prob_a=prob_a,
    prob_b=prob_b,
    post_transform="SOFTMAX"
)
print("NumPy:", numpy_labels.shape, numpy_scores.shape, time.time() - start_time, "seconds")

start_time = time.time()
svm_op.post_transform = "SOFTMAX"
svm_op.prob_a = prob_a
svm_op.prob_b = prob_b
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels shape match:", kp_labels.shape == numpy_labels.shape)
print("Scores shape match:", kp_scores.shape == numpy_scores.shape)
print("Max score error:", np.abs(numpy_scores - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 12: SVC mode with classlabels_strings (triggers integer-index fallback at line 750)
print("Case 12: SVC mode with RBF kernel and classlabels_strings (label index fallback)")
numpy_in = np.random.uniform(-1, 1, (256, 64)).astype(np.float32)
class_count_12 = 3
vectors_per_class_12 = [16, 16, 16]
vector_count_12 = sum(vectors_per_class_12)
support_vectors_12 = np.random.uniform(-1, 1, (vector_count_12, 64)).astype(np.float32).reshape(-1)
coefficients_12 = np.random.uniform(-1, 1, ((class_count_12 - 1), vector_count_12)).astype(np.float32).reshape(-1)
kernel_params_12 = [0.5, 0.0, 0]
rho_12 = [0.1, -0.05, 0.15]
# classlabels_strings not None → line 750 branch: labels_lut_data = [0, 1, 2, ...]
classlabels_strings_12 = ["cat", "dog", "bird"]
# NumPy reference: 使用classlabels_ints=[0,1,2]驱动正确的class_count=3
# ssbo版有classlabels_strings时退化为整数索引[0,1,2]，与此一致
numpy_labels_ref12, numpy_scores_ref12 = onnx_reference_svm_classifier(
    numpy_in,
    classlabels_ints=[0, 1, 2],
    coefficients=coefficients_12,
    kernel_params=kernel_params_12,
    kernel_type="RBF",
    rho=rho_12,
    support_vectors=support_vectors_12,
    vectors_per_class=vectors_per_class_12,
)
start_time = time.time()
svm_op.classlabels_ints = None
svm_op.classlabels_strings = classlabels_strings_12
svm_op.coefficients = coefficients_12
svm_op.kernel_params = kernel_params_12
svm_op.kernel_type = "RBF"
svm_op.post_transform = "NONE"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = rho_12
svm_op.support_vectors = support_vectors_12
svm_op.vectors_per_class = vectors_per_class_12
kp_labels, kp_scores = svm_op.run(numpy_in)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels match (index):", np.array_equal(numpy_labels_ref12, kp_labels))
print("Max score error:", np.abs(numpy_scores_ref12 - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores_ref12, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 13: SVM_LINEAR mode with classlabels_strings (triggers integer-index fallback at line 827)
print("Case 13: SVM_LINEAR mode with classlabels_strings (label index fallback)")
numpy_in_13 = np.random.uniform(-1, 1, (100, 32)).astype(np.float32)
class_count_13 = 3
coefficients_13 = np.random.uniform(-1, 1, (class_count_13, 32)).astype(np.float32).reshape(-1)
rho_13 = [0.5]
classlabels_strings_13 = ["alpha", "beta", "gamma"]
# NumPy reference: 使用classlabels_ints=[0,1,2]驱动class_count=3，与ssbo整数索引退化一致
numpy_labels_ref13, numpy_scores_ref13 = onnx_reference_svm_classifier(
    numpy_in_13,
    classlabels_ints=[0, 1, 2],
    coefficients=coefficients_13,
    kernel_params=None,
    kernel_type="LINEAR",
    rho=rho_13,
    support_vectors=None,
    vectors_per_class=None,
)
start_time = time.time()
svm_op.classlabels_ints = None
svm_op.classlabels_strings = classlabels_strings_13
svm_op.coefficients = coefficients_13
svm_op.kernel_params = None
svm_op.kernel_type = "LINEAR"
svm_op.post_transform = "NONE"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = rho_13
svm_op.support_vectors = None
svm_op.vectors_per_class = None
kp_labels, kp_scores = svm_op.run(numpy_in_13)
print(f"{svm_op}:", kp_labels.shape, kp_scores.shape, time.time() - start_time, "seconds")
print("Labels match (index):", np.array_equal(numpy_labels_ref13, kp_labels))
print("Max score error:", np.abs(numpy_scores_ref13 - kp_scores).max())
print("Scores All close:", np.allclose(numpy_scores_ref13, kp_scores, rtol=1e-4, atol=1e-4))
print()

# Case 14: SVC mode with unsupported kernel type — expects ValueError (line 688-690)
print("Case 14: SVC mode with unsupported kernel type (expects ValueError)")
numpy_in_14 = np.random.uniform(-1, 1, (16, 8)).astype(np.float32)
svm_op.classlabels_ints = [0, 1]
svm_op.classlabels_strings = None
svm_op.coefficients = np.random.uniform(-1, 1, (1, 4)).astype(np.float32).reshape(-1)
svm_op.kernel_params = [0.5, 0.0, 0]
svm_op.kernel_type = "UNKNOWN_KERNEL"
svm_op.post_transform = "NONE"
svm_op.prob_a = None
svm_op.prob_b = None
svm_op.rho = [0.0]
svm_op.support_vectors = np.random.uniform(-1, 1, (4, 8)).astype(np.float32).reshape(-1)
svm_op.vectors_per_class = [2, 2]
try:
    svm_op.run(numpy_in_14)
    print("ERROR: expected ValueError was not raised!")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
print()

