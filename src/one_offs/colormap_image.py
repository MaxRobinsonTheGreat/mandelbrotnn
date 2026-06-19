"""Super-lite: turn an image into a matplotlib-colormapped version of itself.

Loads an image, collapses it to grayscale intensity, then re-colors it through a
matplotlib colormap and saves the result.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'DatasetImages/hands_drawing.png'
out_path = 'DatasetImages/hands_drawing_bone.png'
cmap = 'bone'

# Load as grayscale intensity in [0, 1]
img = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0

plt.imsave(out_path, img, cmap=cmap, vmin=0, vmax=1)
print(f'Saved {out_path} ({cmap} colormap)')
