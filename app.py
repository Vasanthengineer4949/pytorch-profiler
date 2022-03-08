import streamlit as st
from memoryandexecutioncpugpu.profiler_cpu import cpu
from memoryandexecutioncpugpu.profiler_gpu import gpu


opt = st.sidebar.selectbox("Select a page", ["Home","CPU", "GPU", "Next"])

if opt == "Home":
    st.title("PYTORCH PROFILER")
    st.markdown("This is a tutorial for using Pytorch Profiler for finding ResNet-18's execution time and memory consumption")
    
    st.header("Profiler:")
    st.write("Pytorch profiler is a profiler that is used to collect the performance metrics during training and inference.")
    st.write("Its context manager API is used to understand which operators of the model are expensive.")

    st.header("Profiler API:")
    st.markdown("Syntax")
    st.code("torch.profiler.profile(*, activities=None, schedule=None, on_trace_ready=None, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, use_cuda=None)")
    st.markdown("Parameters")
    st.write('''
    activities (iterable) - list of activity groups whether CPU or GPU to profile. torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA. Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA

    schedule (callable) - step(int) value is taken as input and returns a ProfilerAction value which is specifies the actions to be performed by profiler at each step.

    on_trace_ready (callable) - callback function that is called at each step when schedule parameter returns record and save function of ProfilerAction(ProfilerAction.RECORD_AND_SAVE) during the profiling process.

    record_shapes (bool) - save information about operator's input shapes.

    profile_memory (bool) - track tensor memory allocation/deallocation.

    with_stack (bool) - record source information (file and line number) for the ops.

    with_flops (bool) - use formula to estimate the FLOPs (floating point operations) of specific operators (matrix multiplication and 2D convolution).

    with_modules (bool) - record module hierarchy (including function names) corresponding to the callstack of the op.
    ''')


elif opt == "CPU":
    st.title("CPU Memory Consumption and Execution Time Profiler")
    cpu_exec_time, cpu_mem_cons = cpu()
    st.header("Code:")
    st.code('''
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

    return cpu_execution_time, cpu_memory_consumption''')
    st.header("CPU Execution Time:")
    st.write(cpu_exec_time)
    st.header("CPU Memory Consumption:")
    st.write(cpu_mem_cons)
    

elif opt == "GPU":
    st.title("GPU Memory Consumption and Execution Time Profiler")
    gpu_exec_time, gpu_mem_cons = gpu()
    st.header("Code:")
    st.code('''
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
''')
    st.header("GPU Execution Time:")
    st.write(gpu_exec_time)
    st.header("GPU Memory Consumption:")
    st.write(gpu_mem_cons)

elif "Next" in opt:
    st.title("Pytorch Profiler")
    st.markdown("In this tutorial we saw how to use Pytorch Profiler for finding ResNet-18's execution time and memory consumption. In the next tutorial we will analyze the stack traces and also we will see how to integrate Profiler with Tensorboard.")

    






