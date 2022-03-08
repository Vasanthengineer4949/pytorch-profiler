# Import the necessary libraries
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def cpu():
    # Instantiate and create a simple resnet model
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    # Analyze the model's execution time and memory consumption using the profiler
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("model_cpu_inference"):
            model(inputs)

    # Printing the stats of execution time
    cpu_execution_time = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

    #Printing the stats of memory consumption
    cpu_memory_consumption = prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10)

    return cpu_execution_time, cpu_memory_consumption

