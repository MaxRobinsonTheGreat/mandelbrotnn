import matplotlib.pyplot as plt
from src import training, models, dataset, videomaker

def example_train():
    

def example_render():
    plt.imshow(videomaker.generateClassic(3840, 2160), vmin=0, vmax=1, cmap='gray') # 4k render
    # zoom in to useful locations:
    # xmin  xmax  yoffset
    # -1.8  -0.9  0.2       leftmost bulb/tail
    # -0.9  -0.1  0.5       left upper shoulder of main cardioid

def example_train_capture():


if __name__ == "__main__":
    example_render()