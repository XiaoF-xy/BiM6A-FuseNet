# BiM6A-FuseNet Server Environment

推荐 Linux、Python 3.10、NVIDIA GPU 和与驱动匹配的 PyTorch CUDA wheel。CUDA 12.1 服务器可以直接执行：

```bash
cd BiM6A-FuseNet
bash env/create_conda_env_cuda121.sh
conda activate bim6a_fusenet
python env/check_runtime.py
```

若服务器 CUDA 版本不是 12.1，请先把环境脚本中的 PyTorch wheel 地址改为服务器支持的版本。CPU 脚本只适合检查链路，不建议用于完整五折训练。
