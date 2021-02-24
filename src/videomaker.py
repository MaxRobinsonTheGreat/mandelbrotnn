import imageio
from dataset import mandelbrot
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

# import models
# import time

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

# plt.figure()
# plt.imshow(generateClassic(720, 720, 50, -1.5, -1.0, 0.2), vmin=0, vmax=1)
# plt.imshow(generateClassic(1088, 720, 50, -0.8, -0.3, 0.5), vmin=0, vmax=1)
# plt.show()
# -1.8, -0.9, 0.2 Bulb
# -0.9, -0.1, 0.5 shoulder
# plt.imsave("./captures/images/shoulderReal.png", generateClassic(3840, 2160, -0.9, -0.1, 0.5), vmin=0, vmax=1, cmap='gray')
# plt.imsave("./captures/images/bulbReal.png", generateClassic(3840, 2160, -1.8, -0.9, 0.2), vmin=0, vmax=1, cmap='gray')
# plt.imsave("./captures/images/real.png", generateClassic(3840, 2160), vmin=0, vmax=1, cmap='gray')

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

# m = models.Simple(100, 10).cuda()
# m.load_state_dict(torch.load('./models/simple10010.pt'))
# m.eval()
# x = modelGenerate(m, 1920, 1080).squeeze()
# plt.imsave("./captures/images/skipconn20020_2.png", x, vmin=0, vmax=1, cmap='gray')
# plt.imsave("./captures/images/skipconn20010shoulder.png", modelGenerate(m, 3840, 2160, -0.9, -0.1, 0.5).squeeze(), vmin=0, vmax=1, cmap='gray')
# plt.imsave("./captures/images/skipconn20010bulb.png", modelGenerate(m, 3840, 2160, -1.8, -0.9, 0.2).squeeze(), vmin=0, vmax=1, cmap='gray')
# plt.imsave("./captures/images/simple10010full.png", modelGenerate(m, 3840, 2160).squeeze(), vmin=0, vmax=1, cmap='gray')


# plt.figure()
# plt.imshow(modelGenerate(m, 1088, 720, -0.8, -0.3, 0.5), vmin=0, vmax=1)
# plt.imshow(modelGenerate(m, 720, 720, -1.5, -1.0, 0.2), vmin=0, vmax=1)

# plt.show()

# start = time.perf_counter()
# modelGenerate(m, 1088, 720)
# stop = time.perf_counter()
# print(stop-start)

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
