import imageio
from src.dataset import mandelbrot
from tqdm import tqdm
import torch
import numpy as np


def generateClassic(resx, resy, xmin=-2.4, xmax=1, yoffset=0, max_depth=50):
    """ 
    Generates an image of the true mandelbrot set in 2d linear space with a given resolution.\
    Prioritizes resolution over ease of positioning, so the resolution is always preserved\
    and the y range cannot be directly tuned.

    Parameters: 
    resx (int): width of image
    resy (int): height of image
    xmin (float): minimum x value in the 2d space
    xmax (float): maximum x value in the 2d space
    yoffset (float): how much to shift the y position
    max_depth (int): max depth param for mandelbrot function

    Returns: 
    numpy array: 2d float array representing an image 
    """
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
    """ 
    Generates an image of a model's predition of the mandelbrot set in 2d linear\
    space with a given resolution. Prioritizes resolution over ease of positioning,\
    so the resolution is always preserved and the y range cannot be directly tuned.

    Parameters: 
    model (torch.nn.Module): torch model with input size 2 and output size 1
    resx (int): width of image
    resy (int): height of image
    xmin (float): minimum x value in the 2d space
    xmax (float): maximum x value in the 2d space
    yoffset (float): how much to shift the y position
    max_depth (int): max depth param for mandelbrot function

    Returns: 
    numpy array: 2d float array representing an image 
    """
    with torch.no_grad():
        iteration = (xmax-xmin)/resx
        X = torch.arange(xmin, xmax, iteration).cuda()
        y_max = iteration * resy/2
        Y = torch.arange(-y_max-yoffset,  y_max-yoffset, iteration)
        im = []
        # slices each row of the image into batches to be fed into the nn.
        # can be accelerated by putting the entire image in a single batch
        # and resizing, but 4k renders do not fit on my gpu :(
        for y in Y:
            ys = torch.ones(len(X)).cuda() * y
            points = torch.stack([X, ys], 1)
            out = model(points)
            im.append(out)
        im = torch.stack(im, 0)
        im = torch.clamp(im, 0, 1) # doesn't add weird pure white artifacts
        return im.cpu().numpy()


class VideoMaker:
    """ 
    Opens a file writer to begin saving generated model images during training. 
    NOTE: Must call .finish() to close file writer.

    Parameters: 
    filename (string): Name to save the file to 
    fps (int): FPS to save the final mp4 to
    dims (tuple(int, int)): x y resolution to generate images at. For best results,\
        use values divisible by 16.
    capture_rate (int): read by the train() to add a frame after this many batches
    """
    def __init__(self, filename='autosave.mp4', fps=30, dims=(100, 100), capture_rate=10):
        self.writer = imageio.get_writer('./captures/'+filename, fps=fps)
        self.dims=dims
        self.capture_rate=capture_rate

    def generateFrame(self, model):
        """
        Generates a single frame using `modelGenerate` with the given model
        """
        model.eval()
        im = modelGenerate(model, self.dims[0], self.dims[1])
        model.train()
        self.writer.append_data(np.uint8(im*255))

    def finish(self):
        self.writer.close()
