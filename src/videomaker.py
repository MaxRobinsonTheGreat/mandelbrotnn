import os, torch
from src.dataset import mandelbrot, smoothMandelbrot, mandelbrotGPU, mandelbrotTensor, XMIN, XMAX, YMIN, YMAX
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("./captures/images", exist_ok=True)

# Aspect ratio of the canonical view window (width / height). Render grids that want
# the full window with undistorted (square) pixels should size resx:resy to this.
ASPECT = (XMAX - XMIN) / (YMAX - YMIN)


def windowResolution(width, xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX):
    """Given a width, return (resx, resy) whose aspect matches the [xmin,xmax]x[ymin,ymax]
    window, so the full window renders with square pixels. Height is rounded to even
    (libx264/yuv420p require even dimensions)."""
    aspect = (xmax - xmin) / (ymax - ymin)
    h = round(width / aspect)
    h += h % 2
    return width, h

def renderMandelbrot(resx, resy, xmin=XMIN, xmax=XMAX, yoffset=0, max_depth=50, gpu=False, target='smooth', ymin=None, ymax=None, precision=64):
    """
    Generates an image of the true mandelbrot set in 2d linear space with a given resolution.

    Parameters:
    resx (int): width of image
    resy (int): height of image
    xmin (float): minimum x value in the 2d space
    xmax (float): maximum x value in the 2d space
    yoffset (float): how much to shift the y position (legacy framing only)
    max_depth (int): max depth param for mandelbrot function
    target (str): 'smooth' (legacy) or 'periodic' (periodic log-distance target).\
        Should match the target a model was trained on. See src.dataset notes.
    ymin, ymax (float): if both given, the rendered region is exactly\
        [xmin,xmax]x[ymin,ymax] (window-accurate). If omitted, the y range is derived\
        from the resolution around yoffset (legacy framing, used for zoom renders).
    precision (int): GPU compute precision, 64 (default, deep accuracy) or 32\
        (~5x faster, fine for shallow renders).
/
    Returns:
    numpy array: 2d float array representing an image
    """
    if ymin is None or ymax is None:
        step_size = (xmax-xmin)/resx
        y_start = step_size * resy/2
        ymin = -y_start-yoffset
        ymax = y_start-yoffset
    if not gpu:
        X = np.linspace(xmin, xmax, resx)
        Y = np.linspace(ymin, ymax, resy)
        im = np.zeros((resy,resx))
        for j, x in enumerate(tqdm(X)):
            for i, y in enumerate(Y):
                im[i, j] = mandelbrot(x, y, max_depth, target=target, precision=precision)
        return im
    else:
        return mandelbrotGPU(resx, resy, xmin, xmax, ymin, ymax, max_depth, target=target, precision=precision).cpu().numpy()


def renderModel(model, resx, resy, xmin=XMIN, xmax=XMAX, yoffset=0, linspace=None, max_gpu=False, keep_cuda=False, ymin=None, ymax=None, verbose=False):
    """
    Generates an image of a model's prediction of the mandelbrot set in 2d linear\
    space with a given resolution.

    Parameters:
    model (torch.nn.Module): torch model with input size 2 and output size 1
    resx (int): width of image
    resy (int): height of image
    xmin (float): minimum x value in the 2d space
    xmax (float): maximum x value in the 2d space
    yoffset (float): how much to shift the y position (legacy framing only)
    linspace (torch.tensor())): linear space of (x, y) points corresponding to each\
        pixel. Shaped into batches such that shape == (resx, resy, 2) or shape == \
        (resx*resy, 2). Default None, and a new linspace will be generated automatically.
    max_gpu (boolean): if True, the entire linspace will be squeezed into a single batch.
        Requires decent gpu memory size and is significantly faster.
    keep_cuda (boolean): if True, the output and linspace will not be removed from the gpu and will be returned as cuda tensors.
    ymin, ymax (float): if both given, the rendered region is exactly\
        [xmin,xmax]x[ymin,ymax] (window-accurate). If omitted, the y range is derived\
        from the resolution around yoffset (legacy framing, used for zoom renders).

    Returns:
    numpy array: 2d float array representing an image
    OR IF KEEP_CUDA IS TRUE:
    torch.tensor: 2d float tensor representing an image
    """
    with torch.no_grad():
        model.eval()

        if linspace is None:
            linspace = generateLinspace(resx, resy, xmin, xmax, yoffset, ymin, ymax)

        linspace = linspace.cuda()
        
        if not max_gpu:
            # slices each row of the image into batches to be fed into the nn.
            im_slices = []
            if verbose:
                for points in tqdm(linspace):
                    im_slices.append(model(points))
            else:
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
        model.train()
        im = im.squeeze()
        if not keep_cuda:
            im = im.cpu().numpy()
            linspace = linspace.cpu()
        return im



def renderModelWindow(model, width=960, xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX,
                      max_gpu=True, keep_cuda=False, target=None):
    """Render a model over the full [xmin,xmax]x[ymin,ymax] window with square pixels.

    Convenience wrapper around renderModel: picks an aspect-correct (resx, resy) for the
    given width so the whole window is shown undistorted. Use this for training previews."""
    resx, resy = windowResolution(width, xmin, xmax, ymin, ymax)
    return renderModel(model, resx, resy, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                       max_gpu=max_gpu, keep_cuda=keep_cuda)


def generateLinspace(resx, resy, xmin=XMIN, xmax=XMAX, yoffset=0, ymin=None, ymax=None):
    if ymin is not None and ymax is not None:
        # Window-accurate: the grid spans exactly [xmin,xmax]x[ymin,ymax].
        X = torch.linspace(xmin, xmax, resx).cuda()
        Y = torch.linspace(ymin, ymax, resy)
    else:
        # Legacy framing: square pixels, y-extent derived from resolution around yoffset
        # (used by zoom renders where the x-window is specified and y follows).
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
        use values divisible by 16. When `shots` is None the height is adjusted so the\
        full window renders with square pixels (no crop, no stretch).
    capture_rate (int): batches per frame
    window (tuple(xmin, xmax, ymin, ymax)): the view window for the (non-zoom) base\
        framing. Defaults to the canonical dataset window. Ignored when `shots` is set.
    """
    def __init__(self, name='autosave', fps=30, dims=(100, 100), capture_rate=10, shots=None, max_gpu=False, cmap='magma', window=None):
        self.name = name
        self.capture_rate=capture_rate
        self.max_gpu = max_gpu
        self.shots = shots
        self.cmap = cmap
        self.fps = fps
        os.makedirs(f'./frames/{self.name}', exist_ok=True)

        if shots is None:
            # Window-accurate static framing: render the full window with square pixels.
            if window is None:
                window = (XMIN, XMAX, YMIN, YMAX)
            self._xmin, self._xmax, self._ymin, self._ymax = window
            self._yoffset = 0
            resx, resy = windowResolution(dims[0], self._xmin, self._xmax, self._ymin, self._ymax)
            self.dims = (resx, resy)
            if (resx, resy) != tuple(dims):
                print(f"VideoMaker: adjusted dims {tuple(dims)} -> {(resx, resy)} to fit the "
                      f"window with square pixels.")
            self.linspace = generateLinspace(resx, resy, self._xmin, self._xmax,
                                              ymin=self._ymin, ymax=self._ymax)
        else:
            # Legacy zoom framing: y-extent follows the resolution; shots drive the window.
            self.dims = dims
            self._xmin, self._xmax = XMIN, XMAX
            self._ymin = self._ymax = None
            self._yoffset = 0
            self.linspace = generateLinspace(self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset)

        if max_gpu:
            self.linspace = torch.reshape(self.linspace, (self.dims[0]*self.dims[1], 2))

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
