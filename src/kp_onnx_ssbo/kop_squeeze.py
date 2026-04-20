import kp
import numpy as np

class SqueezeOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"SqueezeOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape) if isinstance(inputs[0], np.ndarray) else []))
        if len(inputs) > 1:
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
        input_shape = input_tensors[0][1]
        if len(input_tensors) > 1:
            axes = input_tensors[1][0].data().astype(np.int32).tolist()
            axes = [i if i >= 0 else i + len(input_shape) for i in axes]
            for axis in axes:
                assert 0 <= axis < len(input_shape), f"Axis {axis} is out of bounds for array of dimension {len(input_shape)}）"
                assert input_shape[axis] == 1, f"Cannot select an axis to squeeze out which has size not equal to one"
        else:
            axes = [i for i, d in enumerate(input_shape) if d == 1]

        out_shape = [dim for idx, dim in enumerate(input_shape) if idx not in axes]
        tensor_out = input_tensors[0][0]
        return [(tensor_out, out_shape)]