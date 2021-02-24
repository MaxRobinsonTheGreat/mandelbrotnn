from src.videomaker import generateClassic, modelGenerate, VideoMaker
from src.training import train
from src.dataset import MandelbrotDataSet
from src import models
import matplotlib.pyplot as plt

def example_train():
    model = models.Simple(50, 5)
    dataset = MandelbrotDataSet(50000)
    train(model, dataset, 10) # train for 10 epochs (batch size=1000)

def example_render():
    plt.imshow(videomaker.generateClassic(3840, 2160), vmin=0, vmax=1, cmap='gray') # 4k render
    # zoom into useful locations:
    # xmin  xmax  yoffset
    # -1.8  -0.9  0.2       leftmost bulb/tail
    # -0.9  -0.1  0.5       left upper shoulder of main cardioid

    # you can also load a model and use modelGenerate(model, same params^)

def example_train_capture():
    vidmaker = VideoMaker(dims=(960, 544), capture_rate=5)
    model = models.Simple(50, 5)
    dataset = MandelbrotDataSet(50000)
    train(model, dataset, 10, vm=vidmaker)

if __name__ == "__main__":
    example_render()

# TODO:
# -readme
# -requirements
# -docstrings
# -remove comments
# -examples