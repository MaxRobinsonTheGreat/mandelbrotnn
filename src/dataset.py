import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import math

def m(a, max_depth):
  z = 0
  for n in range(1, max_depth):
    z = z**2 + a
    if abs(z) > 2:
      return min((n-1)/50, 1)
  return 1.0

def mandelbrot(x, y, max_depth=50):
	return m(x + 1j * y, max_depth)

class MandelbrotDataSet(Dataset):
    def __init__(self, size=1000, max_depth=50, xmin=-2.2, xmax=0.7, ymin=-1.1, ymax=1.1):
        self.inputs = []
        self.outputs = []
        print("Generating Dataset")
        for _ in tqdm(range(size)):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            self.inputs.append(torch.tensor([x, y]))
            self.outputs.append(torch.tensor(mandelbrot(x, y, max_depth)))

    def __getitem__(self, i):
      return self.inputs[i], self.outputs[i]

    def __len__(self):
        return len(self.inputs)