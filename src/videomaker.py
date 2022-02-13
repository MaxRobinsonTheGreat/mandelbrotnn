import imageio
from src.dataset import mandelbrot
from tqdm import tqdm
import torch
import numpy as np


def renderMandelbrot(resx, resy, xmin=-2.4, xmax=1, yoffset=0, max_depth=50):
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
    X = np.arange(xmin, xmax, iteration)[:resx]
    y_max = iteration * resy/2
    Y = np.arange(-y_max-yoffset,  y_max-yoffset, iteration)[:resy]
    im = np.zeros((resy,resx))
    for j, x in enumerate(tqdm(X)):
        for i, y in enumerate(Y):
            im[i, j] = mandelbrot(x, y, max_depth)
    return im


def renderModel(model, resx, resy, xmin=-2.4, xmax=1, yoffset=0, linspace=None, max_gpu=False):
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
    linspace (torch.tensor())): linear space of (x, y) points corresponding to each\
        pixel. Shaped into batches such that shape == (resx, resy, 2) or shape == \
        (resx*resy, 2). Default None, and a new linspace will be generated automatically.
    max_gpu (boolean): if True, the entire linspace will be squeezed into a single batch. 
        Requires decent gpu memory size and is significantly faster.

    Returns: 
    numpy array: 2d float array representing an image 
    """
    with torch.no_grad():
        model.eval()
        if linspace is None:
            linspace = generateLinspace(resx, resy, xmin, xmax, yoffset)
        
        linspace = linspace.cuda()
        
        if not max_gpu:
            # slices each row of the image into batches to be fed into the nn.
            im_slices = []
            for points in linspace:
                im_slices.append(model(points))
            im = torch.stack(im_slices, 0)
        else:
            # otherwise cram the entire image in one batch
            if linspace.shape != (resx*resy, 2):
                linspace = torch.reshape(linspace, (resx*resy, 2))
            im = model(linspace).squeeze()
            im = torch.reshape(im, (resy, resx))


        im = torch.clamp(im, 0, 1) # doesn't add weird pure white artifacts
        linspace = linspace.cpu()
        torch.cuda.empty_cache()
        model.train()
        return im.squeeze().cpu().numpy()


def generateLinspace(resx, resy, xmin=-2.4, xmax=1, yoffset=0):
    iteration = (xmax-xmin)/resx
    X = torch.arange(xmin, xmax, iteration).cuda()[:resx]
    y_max = iteration * resy/2
    Y = torch.arange(-y_max-yoffset,  y_max-yoffset, iteration)[:resy]
    linspace = []
    for y in Y:
        ys = torch.ones(len(X)).cuda() * y
        points = torch.stack([X, ys], 1)
        linspace.append(points)
    return torch.stack(linspace, 0)


class VideoMaker:
    """ 
    Opens a file writer to begin saving generated model images during training. 
    NOTE: Must call `.close()` to close file writer.

    Parameters: 
    filename (string): Name to save the file to 
    fps (int): FPS to save the final mp4 to
    dims (tuple(int, int)): x y resolution to generate images at. For best results,\
        use values divisible by 16.
    capture_rate (int): batches per frame
    """
    def __init__(self, filename='autosave.mp4', fps=30, dims=(100, 100), capture_rate=10, shots=None, max_gpu=False):
        self.writer = imageio.get_writer('./captures/'+filename, fps=fps)
        self.dims=dims
        self.capture_rate=capture_rate
        self.max_gpu = max_gpu
        self._xmin = -2.4
        self._xmax = 1
        self._yoffset = 0
        self.shots = shots

        self.linspace = generateLinspace(self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset)
        if max_gpu:
            self.linspace = torch.reshape(self.linspace, (dims[0]*dims[1], 2))

        self.frame_count = 0



    def generateFrame(self, model):
        """
        Generates a single frame using `renderModel` with the given model and appends it to the mp4
        """
        # if self.shots is not None and len(self.shots) > 0 and self.frame_count >= self.shots[0][0]:
        #     shot = self.shots.pop(0)
        #     self._xmin = shot[1]
        #     self._xmax = shot[2]
        #     self._yoffset = shot[3]
        #     self.linspace = generateLinspace(self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset)

        # model.eval()
        im = renderModel(model, self.dims[0], self.dims[1], linspace=self.linspace, max_gpu=self.max_gpu)
        # model.train()
        self.writer.append_data(np.uint8(im*255))
        self.frame_count += 1

    def close(self):
        self.writer.close()
