import network
import torch

net = torch.jit.script(network.TrueNetXS(num_hidden=1).to("cuda")).eval()
torch.jit.save(net, "nano.pt")
