import numpy as np
import kp


class ReshapeOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "ReshapeOp requires two inputs data and target_shape"
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
        assert len(input_tensors) == 2, "ReshapeOp requires two inputs data and target_shape"
        target_shape_list = input_tensors[1][0].data().tolist()
        original_shape = input_tensors[0][1]

        neg_idx, known_prod = -1, 1
        new_shape = []
        for i, dim in enumerate(target_shape_list):
            if dim == -1:
                assert neg_idx == -1, "Only one dimension can be -1"
                neg_idx = i
            else:
                known_prod *= dim
            if dim == 0:
                new_shape.append(original_shape[i])
            else:
                new_shape.append(dim)

        total = np.prod(original_shape)
        if neg_idx != -1:
            new_shape[neg_idx] = total // known_prod
            assert new_shape[neg_idx] * known_prod == total, \
                f"Reshape {new_shape} does not match the total number of elements {total}"

        tensor_out = input_tensors[0][0]
        return [(tensor_out, new_shape)]
