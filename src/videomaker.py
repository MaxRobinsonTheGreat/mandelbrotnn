import imageio
from dataset import mandelbrot
from tqdm import tqdm
import torch
import numpy as np

def generateClassic(resx, resy, xmin=-2.4, xmax=1, yoffset=0, max_depth=50):
    iteration = (xmax-xmin)/resx
    X = np.arange(xmin, xmax, iteration)
    y_max = iteration * resy/2
    Y = np.arange(-y_max-yoffset,  y_max-yoffset, iteration)
    im = np.zeros((resy,resx))
    for j, x in enumerate(tqdm(X)):
        for i, y in enumerate(Y):
            im[i, j] = mandelbrot(x, y, max_depth)
    return im


def modelGenerate(model, resx, resy, xmin=-2.4, xmax=1, yoffset=0):
    with torch.no_grad():
        iteration = (xmax-xmin)/resx
        X = torch.arange(xmin, xmax, iteration).cuda()
        y_max = iteration * resy/2
        Y = torch.arange(-y_max-yoffset,  y_max-yoffset, iteration)
        im = []
        for y in Y:
            ys = torch.ones(len(X)).cuda() * y
            points = torch.stack([X, ys], 1)
            out = model(points)
            im.append(out)
        im = torch.stack(im, 0)
        im = torch.clamp(im, 0, 1) # doesn't add weird pure white artifacts
        return im.cpu().numpy()

class VideoMaker:
    def __init__(self, filename='autosave.mp4', fps=30, dims=(100, 100), capture_rate=10):
        self.writer = imageio.get_writer('./captures/'+filename, fps=fps)
        self.dims=dims
        self.capture_rate=capture_rate

    def generateFrame(self, model):
        model.eval()
        im = modelGenerate(model, self.dims[0], self.dims[1])
        model.train()
        self.writer.append_data(np.uint8(im*255))

    def finish(self):
        self.writer.close()
