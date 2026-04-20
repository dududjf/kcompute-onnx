import numpy as np
import kp


class ShapeOp:
    def __init__(self, manager: kp.Manager, start: int | None = None, end: int | None = None):
        self.manager = manager
        self.start = start
        self.end = end

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ShapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ShapeOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        output = output_shape

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        shape_in = input_tensors[0][1]
        n = len(shape_in)
        start = self.start
        end = self.end

        if start is None:
            start = 0

        def is_end_nan(x):
            return isinstance(x, float) and np.isnan(x)

        if start == 0:
            if end is None or is_end_nan(end):
                ab = None
            elif end < 0:
                ab = (0, n + end)
            else:
                ab = (0, end)
        else:
            if end is None or is_end_nan(end):
                ab = (start, n)
            elif end < 0:
                ab = (start, n + end)
            else:
                ab = (start, end)

        if ab is None:
            tensor_shape = list(shape_in)
        else:
            tensor_shape = list(shape_in[ab[0]:ab[1]])

        tensor_out = input_tensors[0][0]
        return [(tensor_out, tensor_shape)]
