import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
def trace():
    model = models.resnet18().cuda()
    inputs = torch.randn(5, 3, 224, 224).cuda()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
    ) as prof:
        model(inputs)
    prof.export_stacks("D:\Ineuron\Projects\Profiler\stack_trace/tmp\profiler_stacks.txt", "self_cuda_time_total")
    # Print aggregated stats
    return prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5)


trace()