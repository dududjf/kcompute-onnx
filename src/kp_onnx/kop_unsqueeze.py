import kp
import numpy as np

class UnsqueezeOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"UnsqueezeOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        assert len(inputs) == 2, "UnsqueezeOp requires axes input"
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape) if isinstance(inputs[0], np.ndarray) else []))
        numpy_in = np.array(inputs[1], dtype=np.int32).reshape(-1)
        tensor = self.manager.tensor_t(numpy_in, kp.TensorTypes.device)
        input_tensors.append((tensor, list(numpy_in.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]
        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) == 2, "UnsqueezeOp requires axes input"
        input_shape = input_tensors[0][1]
        axes = input_tensors[1][0].data().astype(np.int32).tolist()

        out_rank = len(input_shape) + len(axes)
        axes = sorted(a + out_rank if a < 0 else a for a in axes)
        assert all(0 <= a < out_rank for a in axes), \
            f"Axes {axes} out of range for output rank {out_rank}"

        in_iter = iter(input_shape)
        out_shape = [1 if i in axes else next(in_iter) for i in range(out_rank)]
        tensor_out = input_tensors[0][0]
        return [(tensor_out, out_shape)]