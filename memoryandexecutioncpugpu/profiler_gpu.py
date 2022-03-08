import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def gpu():
    model = models.resnet18().cuda()
    inputs = torch.randn(5, 3, 224, 224).cuda()

    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_gpu_inference"):
            model(inputs)

    gpu_execution_time = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

    gpu_memory_consumption = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)

    return gpu_execution_time, gpu_memory_consumption
