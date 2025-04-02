import torch
from gact.conf import config
from gact.quantizer_split import Quantizer_Split
from gact.quantizer_original import Quantizer_Original
from gact.quantizer_V3 import Quantizer_V3
from gact.quantizer_V5 import Quantizer_V5
from gact.quantizer_V1 import Quantizer_V1
# from gact.quantizer import Quantizer
from gact.quantizer_cache import Quantizer_Cache
from gact.autoprec import AutoPrecision



class Controller:
    def __init__(self, model, env, prefetch_level, keep_last_n=0):
        if not config.compress_activation:
            return

        self.model = model
        
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

        if env == "V3_buffer":
            # 双缓冲测试
            self.quantizer = Quantizer_V3(
                default_bit=default_bit, swap=config.swap, prefetch=config.prefetch, prefetch_level=prefetch_level)
        elif env == "old":
            # 旧测试代码
            self.quantizer = Quantizer_Original(
                default_bit=default_bit, swap=config.swap, prefetch=config.prefetch)
        elif env == "buffer_split":
            self.quantizer = Quantizer_V5(
                default_bit=default_bit, swap=config.swap, prefetch=config.prefetch, prefetch_level=prefetch_level)
        elif env == "new":
            self.quantizer = Quantizer(
                default_bit=default_bit, swap=config.swap, prefetch=config.prefetch, prefetch_level=prefetch_level)
        elif env == "cache":
            self.quantizer = Quantizer_Cache(
                default_bit=default_bit, swap=config.swap, prefetch=config.prefetch, prefetch_level=prefetch_level)
        else:
            print("error")

        # does not quantize model parameters
        self.quantizer.filter_tensors(model.named_parameters())

        self.auto_prec = config.auto_prec
        if self.auto_prec:
            self.ap = AutoPrecision(
                self.model, self.quantizer, config.bit, config.max_bit,
                config.work_dir, config.adapt_interval, config.log_interval)

        self.bit = config.bit
        self.iter = 0

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

    def iterate(self, get_grad):
        if not config.compress_activation:
            return
        self.quantizer.iterate()
        if self.auto_prec:
            self.ap.iterate_wrapper(get_grad)
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
