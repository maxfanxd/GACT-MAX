import torch
from gact.conf import config
from gact.ops import op_quantize, op_dequantize, op_quantize_mask, op_dequantize_mask
from gact.utils import uniform_sample, compute_tensor_bytes


class Quantizer:
    """
    default_bit: the number of bits used to quantize
    swap: if turned on, swap activation memory to CPU
    prefetch: if turned on, activation of the previous layer will be prefetched. the parameter is meaningful only when swap is True
    """

    def __init__(self, default_bit, swap, prefetch, prefetch_level):
        self.unrelated_tensors = set()
        self.default_bit = default_bit
        self.swap = swap
        self.prefetch_level = prefetch_level
        self.current_tid = 0
        if swap:
            self.swap_out_stream = torch.cuda.Stream()
            # 动态创建预取流列表
            self.swap_in_streams = [
                torch.cuda.Stream() 
                for _ in range(prefetch_level)
            ] if prefetch_level > 0 else []
            # self.swap_in_stream_1 = torch.cuda.Stream()  # 用于预取tid-1层
            # self.swap_in_stream_2 = torch.cuda.Stream()  # 新增流用于预取tid-2层
        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.prefetch = prefetch

        # 原来的同步逻辑，会导致过度同步
        # if prefetch:
        #     self.start_prefetch_event = torch.cuda.Event(blocking=True)
        #     self.end_prefetch_event = torch.cuda.Event(blocking=True)
        if self.prefetch and self.swap:
            self.prefetch_events = [(
                torch.cuda.Event(blocking=True),  # start_event
                torch.cuda.Event(blocking=True),  # end_event
            )   for _ in range(self.prefetch_level)
            ]

        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.seeds = {}
        self.bits = {}
        self.dims = {}
        self.iter = 0
        self.seed_iter = 0

    def filter_tensors(self, pairs):
        for _, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    # return should_be_quantized, is_dropout_mask
    # treat dropout mask differently because it can be quantized with 1 bit with a specialized kernel
    def check_quantize(self, input_tensor):
        # does not quantize parameters
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False, False
        # special check for saved mask
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            if (input_tensor.max() == 1) and (input_tensor.min() == 0):
                return True, True
            return False, False
        # only quantize float16 and float32
        if input_tensor.dtype not in [torch.float32, torch.float16]:
            return False, False
        # only quantize activation that requires gradient
        # for example: BN statistics (running mean/var) should not be quantized
        if input_tensor.requires_grad is False:
            return False, False
        # only quantize 2/3/4D tensors for now
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
            ):
            return False, False
        return True, False

    def __del__(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        del self.unrelated_tensors

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.iter += 1

    def generate_tensor_key(self, t, tid):
        if config.check_dup:
            # sample 100 elements data pointer + tensor.sum() as the key
            sample_cnt = min(100, t.numel())
            key = uniform_sample(t, sample_cnt, add_dataptr=True)
            key.append(t.sum().item())
            return tuple(key)
        else:
            return (tid)

    def quantize(self, input):
        quantize, is_dropout_mask = self.check_quantize(input)

        if not quantize:
            return False, input

        # special case: use 1 bit to quantize dropout mask
        if is_dropout_mask:
            q_inputs = op_quantize_mask(input)
            return True, is_dropout_mask, q_inputs

        tid = self.tid
        self.tid += 1
        input_shape = input.shape

        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map

        if not skip_quantize:
            if self.iter == 0:
                bit = self.default_bit
                self.bits[tid] = bit
                self.dims[tid] = input.numel()
                self.seeds[tid] = tid
            else:
                bit = self.bits[tid]
            # quantize
            q_inputs = op_quantize(
                input, bit, self.seeds[tid] + self.seed_iter)
            if self.swap:
                #  with torch.cuda.stream(self.swap_out_stream):
                # self.swap_out_stream.wait_stream(self.compute_stream)
                q_input_cpu = torch.empty(
                    q_inputs[0].shape,
                    dtype=q_inputs[0].dtype,
                    device="cpu",
                    pin_memory=True,
                )
                q_input_cpu.copy_(q_inputs[0], non_blocking=True)
                q_input_gpu = q_inputs[0]
                del q_input_gpu
                q_inputs[0] = q_input_cpu
            self.ptr_qtensor_map[key] = [q_inputs, 1, tid]
        else:
            # increase the ref count
            self.ptr_qtensor_map[key][1] += 1
        return True, is_dropout_mask, key, input_shape, tid

        


    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        is_dropout_mask = input[1]
        if is_dropout_mask:
            _, is_dropout_mask, q_inputs = input
            ret = op_dequantize_mask(q_inputs)
            return ret

        _, _, key, input_shape, tid = input
        q_inputs, ref_cnt, key_tid = self.ptr_qtensor_map[key]

        if self.start_bwd and self.swap:
            # bwd开始时要等待最后一个swap出去的数据
            self.compute_stream.wait_stream(self.swap_out_stream)
            self.start_bwd = False

        if self.prefetch and self.swap:
            # 每次计算的时候要确保针对这一层的prefetch已经结束
            self.prefetch_events[tid % self.prefetch_level][1].wait(self.compute_stream)

        if not q_inputs[0].is_cuda:
            # 需要的数据不在GPU内存上，则需要进行内存交换
            # 一般是最开始，可以理解成一个不需要的同步的流，标号为[tid % self.prefetch_level]
            q_inputs[0] = q_inputs[0].cuda(non_blocking=False)

        if self.prefetch and self.swap:
            # 开始逐渐向前预取，参数化
            for i in range(self.prefetch_level + 1):
                # 假设prefetch_level = 2, 会枚举1, 2，也就是预取tid - 1, tid - 2
                if tid - i < 0:
                    break
                idx = (tid - i) % self.prefetch_level
                self.prefetch_events[idx][0].record()
                with torch.cuda.stream(self.swap_in_streams[idx]):
                    self.prefetch_events[idx][0].wait(self.swap_in_streams[idx])
                    previous_key = self.layer_key_map[tid - i]
                    if previous_key in self.ptr_qtensor_map:
                        q_previous_inputs, _, _ = self.ptr_qtensor_map[previous_key]
                        if not q_previous_inputs[0].is_cuda:
                            q_previous_inputs[0] = q_previous_inputs[0].cuda(non_blocking=True)
                    self.prefetch_events[idx][1].record()  # 预取完成后设置end标记

        ret = op_dequantize(q_inputs, input_shape)

        ref_cnt -= 1
        if ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret