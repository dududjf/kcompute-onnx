import kp
import numpy as np
from .shader_utils import compile_source


class TopKOp:
    def __init__(self, manager: kp.Manager, axis: int = -1, largest: int = 1, sorted: int = 1):
        self.manager = manager
        self.axis = axis
        self.largest = largest
        self.sorted = sorted
        self.shader_source_template = r'''
#version 450
layout (local_size_x = 256) in;
layout (binding = 0) buffer buf_in_data { {DTYPE} in_data[]; };
layout (binding = 1) buffer buf_out_val { {DTYPE} out_val[]; };
layout (binding = 2) buffer buf_out_ind { float out_ind[]; };
layout (constant_id = 0) const float dim_f = 0;
layout (constant_id = 1) const float k_f = 0;
layout (constant_id = 2) const float stride_inner_f = 1; 
layout (constant_id = 3) const float largest_f = 0;
const uint MAX_CAPACITY = 2048; 
const uint SHARED_SIZE = 4096;

shared {DTYPE} s_val[SHARED_SIZE];
shared uint s_ind[SHARED_SIZE];

const {DTYPE} MAX_VAL = {MAX_VAL_LITERAL};
const {DTYPE} MIN_VAL = {MIN_VAL_LITERAL};

void compare_and_swap(uint i, uint j, bool dir) {
    {DTYPE} v_i = s_val[i];
    {DTYPE} v_j = s_val[j];
    uint idx_i = s_ind[i];
    uint idx_j = s_ind[j];

    bool val_gt = (v_i > v_j);
    bool val_lt = (v_i < v_j);
    bool val_eq = (v_i == v_j);
    bool idx_better = (idx_i < idx_j);

    bool key_gt, key_lt;

    if (largest_f != 0.0) { 
        key_gt = val_gt || (val_eq && idx_better);
        key_lt = val_lt || (val_eq && !idx_better);
    } else {
        key_gt = val_gt || (val_eq && !idx_better);
        key_lt = val_lt || (val_eq && idx_better);
    }

    bool swap = false;
    if (dir) { 
        if (key_gt) swap = true;
    } else {
        if (key_lt) swap = true;
    }

    if (swap) {
        s_val[i] = v_j;
        s_val[j] = v_i;
        s_ind[i] = idx_j;
        s_ind[j] = idx_i;
    }
}

void bitonic_sort_generic(uint start, uint n, bool direction) {
    for (uint k_stage = 2; k_stage <= n; k_stage <<= 1) {
        for (uint j_stage = k_stage >> 1; j_stage > 0; j_stage >>= 1) {
            for (uint i = gl_LocalInvocationID.x; i < n; i += gl_WorkGroupSize.x) {
                uint idx = i; 
                uint ixj = idx ^ j_stage;
                if (ixj > idx) {
                     bool dir = ((idx & k_stage) == 0);
                     if (!direction) dir = !dir;
                     compare_and_swap(start + idx, start + ixj, dir);
                }
            }
            barrier();
        }
    }
}

void bitonic_merge(uint n, bool direction) {
    for (uint j_stage = n >> 1; j_stage > 0; j_stage >>= 1) {
        for (uint i = gl_LocalInvocationID.x; i < n; i += gl_WorkGroupSize.x) {
            uint idx = i;
            uint ixj = idx ^ j_stage;
            if (ixj > idx) {
                compare_and_swap(idx, ixj, direction);
            }
        }
        barrier();
    }
}

void main() {
    uint inner_idx = gl_WorkGroupID.x; 
    uint outer_idx = gl_WorkGroupID.y;

    uint dim = uint(dim_f);
    uint k_req = uint(k_f);
    uint inner_stride = uint(stride_inner_f);
    uint largest = uint(largest_f);

    uint input_base = outer_idx * (dim * inner_stride) + inner_idx;
    uint output_base = outer_idx * (k_req * inner_stride) + inner_idx;

    {DTYPE} pad_val_worst = (largest != 0.0) ? MIN_VAL : MAX_VAL;
    bool main_dir = (largest == 0.0);
    bool use_full_sort = (dim <= SHARED_SIZE);

    uint tid = gl_LocalInvocationID.x;
    uint wg_size = gl_WorkGroupSize.x;
    
    uint global_step = wg_size * inner_stride; 
    uint curr_read_ptr = input_base + tid * inner_stride;

    if (use_full_sort) {
        uint capacity = 1;
        while(capacity < dim) capacity <<= 1;

        for (uint i = tid; i < capacity; i += wg_size) {
            if (i < dim) {
                s_val[i] = in_data[curr_read_ptr];
                s_ind[i] = i;
            } else {
                s_val[i] = pad_val_worst;
                s_ind[i] = 4294967295u;
            }
            curr_read_ptr += global_step;
        }
        barrier();

        bitonic_sort_generic(0, capacity, main_dir);

    } else {
        uint capacity = 1;
        while(capacity < k_req) capacity <<= 1;
        if (capacity > MAX_CAPACITY) capacity = MAX_CAPACITY;

        for (uint i = tid; i < capacity; i += wg_size) {
            s_val[i] = pad_val_worst;
            s_ind[i] = 4294967295u; 
        }
        barrier();

        for (uint offset = 0; offset < dim; offset += capacity) {
            uint load_count = min(capacity, dim - offset);

            for (uint i = tid; i < capacity; i += wg_size) {
                uint target_idx = capacity + i;
                if (i < load_count) {
                    s_val[target_idx] = in_data[curr_read_ptr];
                    s_ind[target_idx] = offset + i;
                } else {
                    s_val[target_idx] = pad_val_worst;
                    s_ind[target_idx] = 4294967295u;
                }
                curr_read_ptr += global_step;
            }
            barrier();

            bitonic_sort_generic(capacity, capacity, !main_dir);
            bitonic_merge(2 * capacity, main_dir);
        }
    }
    uint curr_write_ptr = output_base + tid * inner_stride;
    
    for (uint i = tid; i < k_req; i += wg_size) {
        out_val[curr_write_ptr] = s_val[i];
        out_ind[curr_write_ptr] = float(s_ind[i]);
        curr_write_ptr += global_step;
    }
}
'''
        self.shader_source_float = self.shader_source_template.replace("{DTYPE}", "float") \
            .replace("{MAX_VAL_LITERAL}", "3.402823466e+38") \
            .replace("{MIN_VAL_LITERAL}", "-3.402823466e+38")

        self.shader_source_int = self.shader_source_template.replace("{DTYPE}", "int") \
            .replace("{MAX_VAL_LITERAL}", "2147483647") \
            .replace("{MIN_VAL_LITERAL}", "(-2147483647 - 1)")

        self.shader_cache = {}
        self.shader_cache[np.dtype(np.float32).type] = compile_source(self.shader_source_float)
        self.shader_cache[np.dtype(np.int32).type] = compile_source(self.shader_source_int)

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"TopKOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"TopKOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if isinstance(inp, np.ndarray) and np.issubdtype(inp.dtype, np.integer):
                numpy_in = inp.reshape(-1).astype(np.int32)
            else:
                numpy_in = inp.reshape(-1).astype(np.float32)

            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        out_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        if updated_algorithms:
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
            seq.record(kp.OpTensorSyncLocal([t[0] for t in out_tensor_and_shape]))
            seq.eval()

        outputs = []
        if len(out_tensor_and_shape) > 0:
            tensor_val, shape_val = out_tensor_and_shape[0]
            outputs.append(tensor_val.data().reshape(shape_val))

            tensor_ind, shape_ind = out_tensor_and_shape[1]
            outputs.append(tensor_ind.data().reshape(shape_ind))

        for t, _ in input_tensors:
            del t
        del updated_tensors
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_x = input_tensors[0][0]
        shape_x = input_tensors[0][1]
        k_arr = input_tensors[1][0].data()
        k_val = int(k_arr[0])

        rank = len(shape_x)
        axis = self.axis
        if axis < 0:
            axis += rank

        dim = shape_x[axis]
        outer = np.prod(shape_x[:axis]) if axis > 0 else 1
        inner = np.prod(shape_x[axis + 1:]) if axis < rank - 1 else 1

        output_shape = list(shape_x)
        output_shape[axis] = k_val
        total_elements = np.prod(output_shape)

        dtype_x = tensor_x.data().dtype
        dtype_key = dtype_x.type
        if dtype_key not in self.shader_cache:
            dtype_key = np.dtype(np.float32).type

        shader = self.shader_cache[dtype_key]
        tensor_out_val = self.manager.tensor(np.zeros(total_elements))
        tensor_out_ind = self.manager.tensor(np.zeros(total_elements, dtype=np.int64))

        updated_tensors.append(tensor_out_val)
        updated_tensors.append(tensor_out_ind)

        workgroup = (inner, outer, 1)
        params = [dim, k_val, inner, self.largest]

        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, tensor_out_val, tensor_out_ind],
            shader,
            workgroup,
            params,
            []
        ))

        return [(tensor_out_val, output_shape), (tensor_out_ind, output_shape)]
