from src.videomaker import renderMandelbrot, renderModel, VideoMaker
from src.training import train
from src.dataset import MandelbrotDataSet
from src import models
import matplotlib.pyplot as plt
import torch


def example_render():
    image = renderMandelbrot(1920, 1088, max_depth=50) # 304x304 render
    plt.imshow(image, vmin=0, vmax=1, cmap='inferno')
    plt.show()
    # 4k render: 3840, 2160
    # 1080p render: 1920, 1088
    # 960, 544
    # 480, 272

    # pass the following params to renderMandelbrot to zoom into useful locations:
    # xmin  xmax  yoffset
    # -1.8  -0.9  0.2       leftmost bulb/tail
    # -0.9  -0.1  0.5       left upper shoulder of main cardioid
    # -0.52,  0.29,  0.93,  Top anntenea
    # -0.18,  -0.13, 1.033, tiny mandelbrot


def example_train():
    print("Initializing model...")
    
    model = models.Simple(150, 10).cuda() # see src.models for more models

    # show the space before we've learned anything
    plt.imshow(renderModel(model, 600, 600), vmin=0, vmax=1, cmap='inferno')
    plt.show()

    dataset = MandelbrotDataSet(200000) # generate a dataset with 200000 random training points

    train(model, dataset, 10, batch_size=10000, use_scheduler=True) # train for 20 epochs

    # show the space again
    plt.imshow(renderModel(model, 600, 600), cmap='inferno')
    plt.show()


def example_render_model():
    # saves a 4k image
    # model = models.Simple().cuda()
    model = models.SkipConn(200, 30)
    model.load_state_dict(torch.load('./models/sc_200_30_2.pt')) # you need to have a model with this name
    model.cuda()
    plt.imshow(renderModel(model, 1280, 720), vmin=0, vmax=1, cmap='inferno')
    plt.show()


def example_train_capture():
    # we will caputre 480x480 video with new frame every 3 epochs
    shots = [
        (300, -2,  -1.2,  0.),
        (449, -2.5, 1, 0),
        (450, -1.8,  -0.9,  0.2),
        (549, -2.5, 1, 0),
        (550, -0.9,  -0.1,  0.5),
        (700, -2.5, 1, 0),
    ]
    shots=None
    vidmaker = VideoMaker(dims=(1280, 720),  capture_rate=1, shots=shots, max_gpu=True)
    vidmaker = None
 
    model = models.SkipConn(200, 10)
    # model.load_state_dict(torch.load('./models/autosave.pt'))
    # model = models.Simple(300, 30)
    # model = models.Fourier(32, 138, 20)
    # model.usePreprocessing()
    # dataset = MandelbrotDataSet(2000000, max_depth=50)
    dataset = MandelbrotDataSet(loadfile='8mil_inv')
    # dataset = MandelbrotDataSet(loadfile='500k_inv')
    train(model, dataset, 10, batch_size=16000, use_scheduler=True, oversample=0.2, snapshots_every=1, vm=vidmaker)


def create_dataset():
    dataset = MandelbrotDataSet(8000000, max_depth=150)
    dataset.save('8mil_inv')

if __name__ == "__main__":
    # create_dataset()
    example_train_capture()
    # example_render_model()
