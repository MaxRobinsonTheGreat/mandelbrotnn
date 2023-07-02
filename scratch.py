import torch

def fourier2d(x, order):
    features = []
    for n in range(1, order+1):
        for m in range(1, order+1):
            f = torch.tensor([torch.cos(n*x[0])*torch.cos(m*x[1]), torch.cos(n*x[0])*torch.sin(m*x[1]), torch.sin(n*x[0])*torch.cos(m*x[1]), torch.sin(n*x[0])*torch.sin(m*x[1])])
            features.append(f)
    return torch.cat(features)

print(fourier2d(torch.tensor([1, 3]), 10))