import torch
from gact.conf import config
from gact.ops import op_quantize, op_dequantize, op_quantize_mask, op_dequantize_mask
from gact.utils import uniform_sample, compute_tensor_bytes


class Quantizer:
    def __init__(self, default_bit, swap, prefetch):
        self.record_done = False
        self.key_num = 0
        self.key_total = 0

        self.unrelated_tensors = set()
        self.default_bit = default_bit
        self.swap = swap
        if swap:
            self.swap_out_stream = torch.cuda.Stream()
            self.swap_in_stream_1 = torch.cuda.Stream()  # For prefetching tid-1
            self.swap_in_stream_2 = torch.cuda.Stream()  # For prefetching tid-2
            self.swap_out_events = {}  # Track swap out events per tid
        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.prefetch = prefetch
        if prefetch:
            self.start_prefetch_event = torch.cuda.Event(blocking=True)
            self.end_prefetch_event = torch.cuda.Event(blocking=True)
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.seeds = {}
        self.bits = {}
        self.dims = {}
        self.iter = 0
        self.seed_iter = 0

    # ... [其他方法保持原样，如filter_tensors, check_quantize等] ...

    def quantize(self, input):
        quantize, is_dropout_mask = self.check_quantize(input)

        if not quantize:
            return False, input

        if is_dropout_mask:
            q_inputs = op_quantize_mask(input)
            return True, is_dropout_mask, q_inputs

        tid = self.tid
        self.tid += 1
        input_shape = input.shape

        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map

        self.key_num += 1

        if not skip_quantize:
            if self.iter == 0:
                bit = self.default_bit
                self.bits[tid] = bit
                self.dims[tid] = input.numel()
                self.seeds[tid] = tid
            else:
                bit = self.bits[tid]
            q_inputs = op_quantize(input, bit, self.seeds[tid] + self.seed_iter)
            if self.swap:
                # 使用swap_out_stream异步移动到CPU
                with torch.cuda.stream(self.swap_out_stream):
                    if self.record_done and self.key_num + 5 < self.key_total:
                        q_input_cpu = torch.empty(
                            q_inputs[0].shape,
                            dtype=q_inputs[0].dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        q_input_cpu.copy_(q_inputs[0], non_blocking=True)
                        # 记录swap完成事件
                        swap_event = torch.cuda.Event()
                        swap_event.record(stream=self.swap_out_stream)
                        self.swap_out_events[tid] = swap_event
                        q_inputs[0] = q_input_cpu
            self.ptr_qtensor_map[key] = [q_inputs, 1, tid]
        else:
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
            self.record_done = True
            self.compute_stream.wait_stream(self.swap_out_stream)
            self.start_bwd = False
            self.key_total = self.key_num
            self.key_num = 0

        if self.prefetch and self.swap:
            self.end_prefetch_event.wait(self.compute_stream)

        if not q_inputs[0].is_cuda:
            q_inputs[0] = q_inputs[0].cuda(non_blocking=False)

        if self.prefetch and self.swap:
            self.start_prefetch_event.record()
            # 预取tid-1层，使用swap_in_stream_1并等待对应swap_out事件
            with torch.cuda.stream(self.swap_in_stream_1):
                if tid > 0:
                    self.start_prefetch_event.wait(self.swap_in_stream_1)
                    previous_tid = tid - 1
                    swap_event = self.swap_out_events.get(previous_tid)
                    if swap_event:
                        swap_event.wait(stream=self.swap_in_stream_1)
                    previous_key_1 = self.layer_key_map[previous_tid]
                    if previous_key_1 in self.ptr_qtensor_map:
                        q_previous_inputs_1, _, _ = self.ptr_qtensor_map[previous_key_1]
                        if not q_previous_inputs_1[0].is_cuda:
                            # 异步复制到GPU
                            q_previous_inputs_1[0] = q_previous_inputs_1[0].to(
                                device='cuda', non_blocking=True
                            )
                    self.end_prefetch_event.record()

            # 预取tid-2层，使用swap_in_stream_2
            if tid > 1:
                previous_tid_2 = tid - 2
                swap_event_2 = self.swap_out_events.get(previous_tid_2)
                with torch.cuda.stream(self.swap_in_stream_2):
                    if swap_event_2:
                        swap_event_2.wait(stream=self.swap_in_stream_2)
                    previous_key_2 = self.layer_key_map[previous_tid_2]
                    if previous_key_2 in self.ptr_qtensor_map:
                        q_previous_inputs_2, _, _ = self.ptr_qtensor_map[previous_key_2]
                        if not q_previous_inputs_2[0].is_cuda:
                            q_previous_inputs_2[0] = q_previous_inputs_2[0].to(
                                device='cuda', non_blocking=True
                            )

        ret = op_dequantize(q_inputs, input_shape)

        ref_cnt -= 1
        if ref_cnt == 0:
            del self.ptr_qtensor_map[key]
            # 清理相关事件
            if tid in self.swap_out_events:
                del self.swap_out_events[tid]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.iter += 1
        # 清空swap_out事件
        if self.swap:
            self.swap_out_events.clear()

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

    def generate_tensor_key(self, t, tid):
        if config.check_dup:
            # sample 100 elements data pointer + tensor.sum() as the key
            sample_cnt = min(100, t.numel())
            key = uniform_sample(t, sample_cnt, add_dataptr=True)
            key.append(t.sum().item())
            return tuple(key)
        else:
            return (tid)
        
    def compute_tensor_bytes(tensor):
        """
        计算张量占用的字节大小。
        Args:
            tensor (torch.Tensor): 输入的张量。
        Returns:
            int: 张量占用的字节大小。
        """
        return tensor.numel() * tensor.element_size()