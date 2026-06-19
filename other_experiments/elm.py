"""Extreme Learning Machine (ELM) image fitter — fixed by Fable.

Run from the repo root:  python other_experiments/elm.py [image] [hidden]
e.g.                     python other_experiments/elm.py ./DatasetImages/smiley_small.png 8000
"""

import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append('.')
from src.imageDataset import ImageDataset

IMAGE = sys.argv[1] if len(sys.argv) > 1 else './DatasetImages/smiley_small.png'
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
W_SCALE = 50.0     # weight init range: bigger = sharper tanh ridges = finer detail
                   # (swept 12/25/50/100/200 on smiley_small at hidden=8000: 50 best)
RIDGE = 1e-8       # relative ridge regularization (scaled by mean diag of H^T H)
SOLVE_CHUNK = 65536


class ELM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, w_scale=W_SCALE):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        self.activation = nn.Tanh()
        # Random frozen features: weights symmetric around zero; biases spread on
        # the same scale so the tanh transitions tile the [-1,1]^2 input domain.
        nn.init.uniform_(self.hidden_layer.weight, -w_scale, w_scale)
        nn.init.uniform_(self.hidden_layer.bias, -w_scale, w_scale)
        for p in self.hidden_layer.parameters():
            p.requires_grad = False

    def features(self, x):
        return self.activation(self.hidden_layer(x))

    def forward(self, x):
        return self.output_layer(self.features(x))


def image_tensors(dataset):
    """All (coords, values) of an ImageDataset, vectorized (matches __getitem__)."""
    h, w = dataset.height, dataset.width
    row, col = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    x = torch.stack([col.flatten() / (w / 2) - 1,
                     row.flatten() / (h / 2) - 1], dim=1).float()
    y = dataset.image[0].reshape(-1, 1).float()
    return x, y


def fit_elm(model, x, y, ridge=RIDGE, chunk=SOLVE_CHUNK):
    """Closed-form ridge solve for the output weights, fp64 on the data's device.

    Normal equations accumulated in chunks: (H^T H + lam*I) beta = H^T y.
    """
    device = x.device
    Hd = model.hidden_layer.weight.shape[0]
    HtH = torch.zeros(Hd, Hd, dtype=torch.float64, device=device)
    Hty = torch.zeros(Hd, y.shape[1], dtype=torch.float64, device=device)
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            Hc = model.features(x[i:i + chunk]).double()
            HtH += Hc.T @ Hc
            Hty += Hc.T @ y[i:i + chunk].double()
        lam = ridge * HtH.diagonal().mean()
        HtH.diagonal().add_(lam)
        beta = torch.linalg.solve(HtH, Hty)            # (Hd, out)
        model.output_layer.weight.data = beta.T.to(model.output_layer.weight.dtype)


@torch.no_grad()
def evaluate(model, x, y, chunk=SOLVE_CHUNK):
    se = 0.0
    for i in range(0, x.shape[0], chunk):
        pred = model(x[i:i + chunk])
        se += ((pred - y[i:i + chunk]) ** 2).sum().item()
    return se / x.shape[0]


@torch.no_grad()
def render(model, height, width, chunk=SOLVE_CHUNK, device='cuda'):
    """Predict every pixel on the dataset's own coordinate grid -> (H, W) array."""
    row, col = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    coords = torch.stack([col.flatten() / (width / 2) - 1,
                          row.flatten() / (height / 2) - 1], dim=1).float().to(device)
    out = torch.cat([model(coords[i:i + chunk]) for i in range(0, coords.shape[0], chunk)])
    return out.reshape(height, width).clamp(0, 1).cpu().numpy()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ImageDataset(IMAGE)
    x, y = image_tensors(dataset)
    x, y = x.to(device), y.to(device)
    print(f"image: {IMAGE} ({dataset.width}x{dataset.height}, {x.shape[0]} px), "
          f"hidden: {HIDDEN}, device: {device}")

    model = ELM(2, HIDDEN, 1).to(device)
    print(f"before solve, MSE: {evaluate(model, x, y):.6f}")
    fit_elm(model, x, y)
    print(f"after solve,  MSE: {evaluate(model, x, y):.6f}")

    img = render(model, dataset.height, dataset.width, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(dataset.image[0].numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[0].set_title('target')
    axes[1].imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1].set_title(f'ELM ({HIDDEN} random features)')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./captures/elm_image_fit.png', dpi=120)
    print("saved ./captures/elm_image_fit.png")
    plt.show()
