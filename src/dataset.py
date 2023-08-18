import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

device="cuda"

# function helper, don't directly call
def _m(a, max_depth):
    z = 0
    for n in range(max_depth):
        z = z**2 + a
        if abs(z) > 2:
            # return 0
            # return min((n-1)/max_depth, 1)
            # return -(math.cos((n-1)*math.pi/50)/2) + (1/2)
            # return math.log(n/10 + 1)
            return smoothMandelbrot(n)
    return 1.0

def smoothMandelbrot(iters, smoothness=50):
    return 1-(1/((iters/smoothness) + 1))

def mandelbrot(x, y, max_depth=50):
    """ 
    Calculates whether the given point is in the mandelbrot set.

    Parameters: 
    x (float): real part of the number
    y (float): complex part of the number
    max_depth (int): Maximum number of recursive steps before deciding\
    whether the value is in the mandelbrot set


    Returns: 
    float: Number between 1 and 0 where 1.0 is in the mandelbrot set and\
    values closer to 1.0 required more steps to determine this
    """
    return _m(x + 1j * y, max_depth)

def mandelbrotGPU(resx, resy, xmin, xmax, ymin, ymax, max_depth):
    X = torch.linspace(xmin, xmax, resx, device=device, dtype=torch.float64)
    Y = torch.linspace(ymin, ymax, resy, device=device, dtype=torch.float64)

    # Create the meshgrid using real and imaginary ranges
    imag_values, real_values = torch.meshgrid(Y, X)

    return mandelbrotTensor(imag_values, real_values, max_depth)

def mandelbrotTensor(imag_values, real_values, max_depth):
    # Combine real and imaginary parts into a complex tensor
    c = real_values + 1j * imag_values

    z = torch.zeros_like(c, dtype=torch.float64, device=device)

    mask = torch.ones_like(z, dtype=torch.bool, device=device)

    final_image = torch.zeros_like(z, dtype=torch.float64, device=device)
    
    for n in range(max_depth):
        z = z**2 + c
        escaped = torch.abs(z) > 2
        mask = ~escaped & mask
        # print(n, smoothMandelbrot(n), torch.tensor([smoothMandelbrot(n)], dtype=torch.float64).cuda().cpu().numpy()[0])
        final_image[mask] = smoothMandelbrot(n)

    # iteration_count = torch.where(iteration_count==0, max_depth, iteration_count)
    final_image[torch.abs(z) <= 2] = 1.0 # all points that never escaped should be set to full white
    return final_image


class MandelbrotDataSet(Dataset):
    """ 
    Creates a dataset of randomized points and their calculated mandelbrot values.
  
    Parameters: 
    size (int): number of randomized points to generate
    max_depth (int): Maximum number of recursive steps before deciding\
      whether the value is in the mandelbrot set
    xmin (float): minimum x value for points
    xmax (float): maximum x value for points
    ymin (float): minimum y value for points
    ymax (float): maximum y value for points
    """
    def __init__(self, size=1000, loadfile=None, max_depth=50, xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1,  dtype=torch.float32, gpu=False):
        self.inputs = []
        self.outputs = []
        if loadfile is not None:
            self.load(loadfile)
        else:
            print("Generating Dataset")
            if not gpu:
                for _ in tqdm(range(size)):
                    x = random.uniform(xmin, xmax)
                    y = random.uniform(ymin, ymax)
                    self.inputs.append(torch.tensor([x, y]))
                    self.outputs.append(torch.tensor(mandelbrot(x, y, max_depth)))
                self.inputs = torch.stack(self.inputs)
                self.outputs = torch.stack(self.outputs)
            else:
                X = (xmin - xmax) * torch.rand((size), dtype=dtype, device=device) + xmax
                Y = (ymin - ymax) * torch.rand((size), dtype=dtype, device=device) + xmax
                self.inputs = torch.stack([X, Y], dim=1).cpu()
                self.outputs = mandelbrotTensor(Y, X, max_depth).cpu()

        self.start_oversample(len(self.inputs))


    def __getitem__(self, i):
        if (i >= len(self.inputs)):
            ind = self.oversample_indices[i - len(self.inputs)]
            return self.inputs[ind], self.outputs[ind], ind.item()
        return self.inputs[i], self.outputs[i], i


    def __len__(self):
        return len(self.inputs) + len(self.oversample_indices)

    def start_oversample(self, max_size):
        self.max_size = max_size
        self.oversample_indices = torch.tensor([], dtype=torch.long)
        self.oversample_buffer = torch.tensor([], dtype=torch.long)
    
    def update_oversample(self):
        self.oversample_indices = self.oversample_buffer[:self.max_size]
        self.oversample_buffer = torch.tensor([], dtype=torch.long)

    def add_oversample(self, indices):
        indices = indices[indices < len(self.inputs)] # remove duplicates
        self.oversample_buffer = torch.cat([self.oversample_buffer, indices], 0)

    def save(self, filename):
        import os
        os.makedirs("./data", exist_ok=True)
        torch.save(self.inputs, './data/'+filename+'_inputs.pt')
        torch.save(self.outputs, './data/'+filename+'_outputs.pt')

    def load(self, filename):
        self.inputs = torch.load('./data/'+filename+'_inputs.pt')
        self.outputs = torch.load('./data/'+filename+'_outputs.pt')
