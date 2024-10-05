import torch
import time
import network

batch_size = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iterations = 100

x = torch.rand([batch_size, 21, 8, 8]).to(device)

model = torch.jit.script(
    network.TrueNet(
        num_resBlocks=8,
        num_hidden=64,
        head_channel_policy=8,
        head_channel_values=4,
    ).to(device)
)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

model.eval()

with torch.no_grad():
    for _ in range(10):
        _ = model(x)

start_time = time.time()
for _ in range(iterations):
    _ = model(x)
end_time = time.time()


elapsed_time = end_time - start_time
evaluations_per_second = (iterations * batch_size) / elapsed_time

print(f"Evaluations per second: {evaluations_per_second}")
