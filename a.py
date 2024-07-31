import torch
import time

batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iterations = 100

x = torch.rand([batch_size, 21, 8, 8]).to(device)

model = torch.jit.load("tz_6515.pt").to(device)

with torch.no_grad():
    for _ in range(10):
        _ = model(x)


start_time = time.time()
with torch.no_grad():
    torch.cuda.synchronize(device=device)
    for _ in range(iterations):
        _ = model(x)
end_time = time.time()


elapsed_time = end_time - start_time
evaluations_per_second = iterations * batch_size / elapsed_time

print(f"Evaluations per second: {evaluations_per_second}")
