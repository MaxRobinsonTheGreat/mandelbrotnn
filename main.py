from src.videomaker import generateClassic, modelGenerate, VideoMaker
from src.training import train
from src.dataset import MandelbrotDataSet
from src import models
import matplotlib.pyplot as plt

import torch


def example_train():
    model = models.Simple()
    dataset = MandelbrotDataSet(100000)
    train(model, dataset, 150, batch_size=10000, use_scheduler=True)
    plt.imshow(modelGenerate(model, 1920, 1088), vmin=0, vmax=1)
    plt.show()


def example_train_capture():
    # save training capture to captures/autosave.mp4
    vidmaker = VideoMaker(dims=(960, 544), capture_rate=5)
    model = models.SkipConn(50, 5)
    dataset = MandelbrotDataSet(50000)
    train(model, dataset, 5, vm=vidmaker)


def example_render():
    plt.imshow(generateClassic(304, 304), vmin=0, vmax=1, cmap='gray') # 304x304 render
    plt.show()
    # 4k render: 3840, 2160
    # 1080p render: 1920, 1088

    # zoom into useful locations:
    # xmin  xmax  yoffset
    # -1.8  -0.9  0.2       leftmost bulb/tail
    # -0.9  -0.1  0.5       left upper shoulder of main cardioid
    

def example_render_model():
    # saves a 4k image
    model = models.Simple().cuda()
    model.load_state_dict(torch.load('./models/autosave.pt'))
    plt.imsave("./captures/images/render.png", modelGenerate(model, 3840, 2160).squeeze(), vmin=0, vmax=1, cmap='gray')


if __name__ == "__main__":
    example_render()
