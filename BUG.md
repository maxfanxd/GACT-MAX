# 这里列举了配置环境容易遇到的BUG
## 缺少libffi.so.6文件
> ImportError: libffi.so.6: cannot open shared object file: No such file or directory

1. sudo find / -name "libffi*"
2. export LD_LIBRARY_PATH=/opt/conda/pkgs/libffi-3.3-he6710b0_2/lib:$LD_LIBRARY_PATH

先找到对应的文件，或者其他版本，然后重新设置环境变量

## 缺少libtorch_cuda_cu.so
> ImportError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory

1. sudo find / -name "libtorch_cuda_cpp*"
2. export LD_LIBRARY_PATH=/opt/conda/envs/pytorch-2.1.1/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

## 装载失败
```
ImportError: /opt/conda/envs/pytorch-2.1.1/lib/python3.11/site-packages/torch/lib/libtorch_python.so: undefined symbol: PyObject_CallOneArg
Traceback (most recent call last):
  File "/home/u210110632/jupyterlab/GPT/gpt-2-700m/code/train.py", line 2, in <module>
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
  File "/home/u210110632/jupyterlab/Python-3.9.17/llm/lib/python3.9/site-packages/transformers/__init__.py", line 26, in <module>
    from . import dependency_versions_check
  File "/home/u210110632/jupyterlab/Python-3.9.17/llm/lib/python3.9/site-packages/transformers/dependency_versions_check.py", line 16, in <module>
    from .utils.versions import require_version, require_version_core
  File "/home/u210110632/jupyterlab/Python-3.9.17/llm/lib/python3.9/site-packages/transformers/utils/__init__.py", line 27, in <module>
    from .chat_template_utils import DocstringParsingException, TypeHintParsingException, get_json_schema
  File "/home/u210110632/jupyterlab/Python-3.9.17/llm/lib/python3.9/site-packages/transformers/utils/chat_template_utils.py", line 39, in <module>
    from torch import Tensor
  File "/home/u210110632/jupyterlab/Python-3.9.17/llm/lib/python3.9/site-packages/torch/__init__.py", line 197, in <module>
    from torch._C import *  # noqa: F403
```

这个说明Pytorch和Python的版本不一致导致的

## #include <ATen/CUDAGeneratorImpl.h>错误
进入GACT/gact/gact/cpp_extension/quantization_cuda_kernel.cu:6:10中，将#include <ATen/CUDAGeneratorImpl.h>改成#include <ATen/cuda/CUDAGeneratorImpl.h>


# 从零开始配虚拟环境

(HAI远程环境) conda init bash
conda create -n llm python=3.9.17 -y
conda activate llm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
然后进gact文件夹安装
pip install -v -e .
然后感觉还是要降级就
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch -c conda-forge
or
conda install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
接下来安装低版本的transformers
pip install transformers==4.12.0
or（前者证明不太行）
pip install transformers==4.20.1 tokenizers==0.11.1

如果出现了error: can't find Rust compiler
则pip install --upgrade pip，然后安装rust：curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh，之后输入1回车，然后source $HOME/.cargo/env，用rustc --version验证是否安装成功，然后重新安装

如果枚举值出问题直接进去跟着提示改枚举值即可

llama的训练中需要：
pip install trl==0.12.2
pip install bitsandbytes

如果要添加清华源：
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --set show_channel_urls yes