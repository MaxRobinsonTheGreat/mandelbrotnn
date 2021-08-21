from src.videomaker import renderMandelbrot, renderModel, VideoMaker
from src.training import train
from src.dataset import MandelbrotDataSet
from src import models
import matplotlib.pyplot as plt


def example_render():
    image = renderMandelbrot(304, 304, yoffset=0, max_depth=100) # 304x304 render
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
    model = models.Simple().cuda()
    # model.load_state_dict(torch.load('./models/autosave.pt')) # you need to have a model with this name
    plt.imsave("./captures/images/render.png", renderModel(model, 3840, 2160), vmin=0, vmax=1, cmap='gray')


def example_train_capture():
    # we will caputre 480x480 video with new frame every 3 epochs
    vidmaker = VideoMaker(dims=(480, 480), capture_rate=3)

    model = models.Simple()
    dataset = MandelbrotDataSet(100000)
    train(model, dataset, 10, batch_size=8000, vm=vidmaker)


if __name__ == "__main__":
    example_train_capture()
