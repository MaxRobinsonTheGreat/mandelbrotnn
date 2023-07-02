import imageio, os, torch
from src.dataset import mandelbrot, smoothMandelbrot, mandelbrotGPU, mandelbrotTensor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("./captures/images", exist_ok=True)

def renderMandelbrot(resx, resy, xmin=-2.4, xmax=1, yoffset=0, max_depth=50, gpu=False):
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
/
    Returns: 
    numpy array: 2d float array representing an image 
    """
    step_size = (xmax-xmin)/resx
    y_start = step_size * resy/2
    ymin = -y_start-yoffset
    ymax = y_start-yoffset
    if not gpu:
        X = np.arange(xmin, xmax, step_size)[:resx]
        Y = np.arange(ymin,  ymax, step_size)[:resy]
        im = np.zeros((resy,resx))
        for j, x in enumerate(tqdm(X)):
            for i, y in enumerate(Y):
                im[i, j] = mandelbrot(x, y, max_depth)
        return im
    else:
        return mandelbrotGPU(resx, resy, xmin, xmax, ymin, ymax, max_depth).cpu().numpy()


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
    # X = torch.linspace(xmin, xmax, resx, device='cuda', dtype=dtype)
    # Y = torch.linspace(-y_max-yoffset,  y_max-yoffset, resy, device='cuda', dtype=dtype)

	# # Create the meshgrid using real and imaginary ranges
    # grid = torch.stack(torch.meshgrid(Y, X), -1)
    # return grid.view(-1, 2)


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
    def __init__(self, name='autosave', fps=30, dims=(100, 100), capture_rate=10, shots=None, max_gpu=False, cmap='magma'):
        self.name = name
        self.dims=dims
        self.capture_rate=capture_rate
        self.max_gpu = max_gpu
        self._xmin = -2.4
        self._xmax = 1
        self._yoffset = 0
        self.shots = shots
        self.cmap = cmap
        self.fps = fps
        os.makedirs(f'./frames/{self.name}', exist_ok=True)

        self.linspace = generateLinspace(self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset)
        if max_gpu:
            self.linspace = torch.reshape(self.linspace, (dims[0]*dims[1], 2))

        self.frame_count = 0

    def generateFrame(self, model):
        """
        Generates a single frame using `renderModel` with the given model and appends it to the mp4
        """
        if self.shots is not None and len(self.shots) > 0 and self.frame_count >= self.shots[0]['frame']:
            shot = self.shots.pop(0)
            self._xmin = shot["xmin"]
            self._xmax = shot["xmax"]
            self._yoffset = shot["yoffset"]
            if len(shot) > 4:
                self.capture_rate=shot["capture_rate"]
            self.linspace = generateLinspace(self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset)

        # model.eval()
        im = renderModel(model, self.dims[0], self.dims[1], linspace=self.linspace, max_gpu=self.max_gpu)
        plt.imsave(f'./frames/{self.name}/{self.frame_count:05d}.png', im, cmap=self.cmap)
        self.frame_count += 1

    def generateVideo(self):
        os.system(f'ffmpeg -y -r {self.fps} -i ./frames/{self.name}/%05d.png -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p ./frames/{self.name}/{self.name}.mp4')
