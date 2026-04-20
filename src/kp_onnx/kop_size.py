import numpy as np
import kp


class SizeOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SizeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SizeOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) >= 1, "SizeOp requires at least one input tensor"
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape) if isinstance(inputs[0], np.ndarray) else []))

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
        assert len(input_tensors) >= 1, "SizeOp requires at least one input tensor"

        shape = input_tensors[0][1]
        total_size = np.prod(shape, dtype=np.int64)
        arr = np.array([total_size], dtype=np.int64)
        tensor_out = self.manager.tensor(arr)
        return [(tensor_out, [])]
