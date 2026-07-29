import torch
import astra

print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print("Test ASTRA CUDA active:", astra.use_cuda())