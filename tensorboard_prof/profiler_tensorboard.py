# Pytorch Profiler with Tensorboard for ResNet-18 tutorial

'''
This is a tutorial for using Pytorch Profiler with Tensorboard for ResNet-18. 
Pytorch profiler is a profiler that is used to measure the time and memory consumption
'''

# Import the necessary packages
import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Prepare the dataset: Load the dataset and perform the required transformations
transform = transforms.Compose(
    [transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# Load the model, create the loss function and optimizer and set it to cuda
device = torch.device("cpu")
model = models.resnet18(pretrained=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()

# Defining the training step function
def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Creating the profiler to record the time and memory consumption for the execution of each event
'''Here wait means that for the first second of the execution, the profiler will not 
record the time and memory it will be warming up until then. After then it will record for the 
next three iterations. This cycle will be repeated twice. In tensorboard, a cycle is said
to be a span'''

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break
        train(batch_data)
        prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.

    

