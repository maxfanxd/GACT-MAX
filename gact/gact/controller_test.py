import torch
from gact.conf import config
from gact.quantizer_split import Quantizer_Split
from gact.quantizer_original import Quantizer_Original
from gact.quantizer_V3 import Quantizer_V3
from gact.quantizer_V5 import Quantizer_V5
from gact.quantizer_V1 import Quantizer_V1
# from gact.quantizer import Quantizer
# from gact.quantizer_cache import Quantizer_Cache
from gact.autoprec import AutoPrecision
import time
import sys



class Controller_Test:
    def __init__(self, model, env, prefetch_level, output, keep_last_n=0, start_level=1, end_level=3):
        if not config.compress_activation:
            return

        self.model = model
        self.output = output
        
        assert(config.bit <= 16 and config.bit > 1)
        if config.bit == 2:
            default_bit = 2
        elif config.bit <= 4:
            default_bit = 4
        elif config.bit <= 8:
            default_bit = 8
        else:
            assert(config.bit <= 16)
            default_bit = 8

        ### 针对prefetch_level的step_time的测试，prefetch_level默认在1-6之间选择(两个参数决定)
        assert(start_level > 0 and start_level < end_level)
        self.start_level = start_level
        self.end_level = end_level
        self.quantizers = {}
        self.step_times = {}
        self.level = start_level
        for level in range(start_level, end_level+1):
            self.quantizers[level] = self.quantizer = Quantizer_V5(
                default_bit=default_bit, swap=config.swap, prefetch=config.prefetch, prefetch_level=level)
            self.step_times[level] = [] # 初始化空列表
        self.quantizer = self.quantizers[level]

        # does not quantize model parameters
        self.quantizer.filter_tensors(model.named_parameters())

        self.auto_prec = config.auto_prec
        if self.auto_prec:
            self.ap = AutoPrecision(
                self.model, self.quantizer, config.bit, config.max_bit,
                config.work_dir, config.adapt_interval, config.log_interval)

        self.bit = config.bit
        self.iter = 0

        self.record_done = False

        
        

    def __del__(self):
        self.uninstall_hook()

    def install_hook(self):
        def pack_hook(x):
            r = self.quantize(x)
            del x
            return r

        def unpack_hook(x):
            r = self.dequantize(x)
            del x
            return r

        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._register_saved_tensors_default_hooks(
                pack_hook, unpack_hook)
        else:
            torch._C._autograd._push_saved_tensors_default_hooks(
                pack_hook, unpack_hook)

    def uninstall_hook(self):
        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._reset_saved_tensors_default_hooks()
        else:
            torch._C._autograd._pop_saved_tensors_default_hooks()

    def start_record(self):
        if not self.record_done and self.iter > 10:
            self.start_time = time.time()

    def iterate(self, get_grad):
        if not config.compress_activation:
            return
        self.quantizer.iterate()
        if self.auto_prec:
            self.ap.iterate_wrapper(get_grad)
        
        if not self.record_done and self.iter > 10:
            print(self.level)
            print(self.step_times[self.level])
            iter_time = time.time() - self.start_time
            self.step_times[self.level].append(iter_time)
            if (self.iter - 10) % 5 == 0:
                self.level = self.level + 1
                if self.level > self.end_level:
                    self.record_done = True
                    # 计算最佳prefetch_level，为了公平起见，把每个前两个删除掉
                    avg_times = [(sum(self.step_times[i][2:])/(len(self.step_times[i])-2)) for i in range(self.start_level, self.end_level + 1)]
                    # 打印每个 level 对应的平均时间
                    print("每个 level 对应的平均时间:")
                    for i, avg_time in enumerate(avg_times, start=self.start_level):
                        print(f"Level {i}: {avg_time}")
                    # 找到最少时间的 level
                    min_avg_time = min(avg_times)
                    min_level = avg_times.index(min_avg_time) + self.start_level
                    print(f"最少时间的 level 是: Level {min_level}，时间为: {min_avg_time}")
                    # 将结果写入文件
                    with open(self.output, 'w') as f:
                        f.write("每个 level 对应的平均时间:\n")
                        for i, avg_time in enumerate(avg_times, start=self.start_level):
                            f.write(f"Level {i}: {avg_time}\n")
                        f.write(f"最少时间的 level 是: Level {min_level}，时间为: {min_avg_time}\n")

                    sys.exit(1)
                self.quantizer = self.quantizers[self.level]

        self.iter += 1
        self.quantizer.seed_iter = self.iter
        


    def quantize(self, input):
        if not config.compress_activation:
            if config.swap:
                # swap original tensor to cpu
                tensor_cpu = torch.empty(
                    input.shape, dtype=input.dtype, device='cpu', pin_memory=True)
                tensor_cpu.copy_(input, non_blocking=True)
                return tensor_cpu
            else:
                return input
        return self.quantizer.quantize(input)

    def dequantize(self, input):
        if not config.compress_activation:
            if config.swap:
                input = input.cuda(non_blocking=True)
            return input
        return self.quantizer.dequantize(input)
