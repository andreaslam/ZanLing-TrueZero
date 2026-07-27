import time

import torch

import network

batch_size = 2048
if torch.cuda.is_available():
    device = "cuda"
elif (
    getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
):
    device = "mps"
else:
    device = "cpu"
print(f"Using: {device}")
iterations = 100
warmups = 10
x_warmup = torch.rand([warmups, batch_size, 21, 8, 8]).to(device)
x_test = torch.rand([iterations, batch_size, 21, 8, 8]).to(device)

model = torch.jit.script(network.TrueNet().to(device))

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

model.eval()

with torch.no_grad():
    for xi, _ in zip(x_warmup, range(warmups)):
        y_ = model(xi)

start_time = time.time()
for xi, _ in zip(x_test, range(iterations)):
    y_ = model(xi)
end_time = time.time()


elapsed_time = end_time - start_time
evaluations_per_second = (iterations * batch_size) / elapsed_time

print(f"Evaluations per second: {evaluations_per_second}")
