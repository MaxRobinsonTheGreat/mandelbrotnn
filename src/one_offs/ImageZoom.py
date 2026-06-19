"""Zoom into a fixed point of a raster image.

Companion to ModelZoom.py: where the model zoom reveals a smooth, infinitely
detailed function, this image zoom reveals that the source is just a finite grid
of pixels. As we push in, nearest-neighbour sampling makes each source pixel grow
into a clean, crisp block (no blur, no interpolation) so the pixel grid is obvious.
"""
import os
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- config ---------------------------------------------------------------
image_path = 'DatasetImages/hands_drawing.png'
video_name = 'image_zoom_hands'
cmap = 'bone'  # set to None to keep the image's own RGB colors

# Point to zoom into, as a fraction of the image (0,0)=top-left, (1,1)=bottom-right.
center = (0.31, 0.5)

resx, resy = 960, 832          # output frame resolution
frames = 400                   # number of frames
zoom_factor = 0.012            # fraction the half-window shrinks each frame (exponential)
final_resx, final_resy = 1920, 1664
fps = 60
# --------------------------------------------------------------------------

# Load as grayscale intensity in [0, 1] when colormapping, else keep RGB.
if cmap is not None:
    img = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
else:
    img = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32) / 255.0

src_h, src_w = img.shape[:2]
aspect = resx / resy

# Start centered on the whole image; pan toward the target point as we zoom in.
img_cx, img_cy = src_w / 2, src_h / 2
tgt_cx, tgt_cy = center[0] * src_w, center[1] * src_h

# Initial half-window: the whole image (matched to the output aspect ratio).
half_w0 = src_w / 2
half_h0 = half_w0 / aspect
if half_h0 > src_h / 2:  # image is taller than the frame aspect; fit by height instead
    half_h0 = src_h / 2
    half_w0 = half_h0 * aspect


def sample(cx, cy, half_w, half_h, out_w, out_h):
    """Nearest-neighbour crop+resample of the window centered on (cx, cy)."""
    xs = np.linspace(cx - half_w, cx + half_w, out_w)
    ys = np.linspace(cy - half_h, cy + half_h, out_h)
    # round to the nearest source pixel -> identical neighbours form crisp blocks
    xi = np.clip(np.round(xs).astype(int), 0, src_w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, src_h - 1)
    return img[np.ix_(yi, xi)]


def center_for(half_w):
    """Pan from the image center to the target, tied to the zoom level so the
    target glides into place at a constant relative screen position."""
    p = 1 - half_w / half_w0  # 0 at full view, ->1 as we zoom all the way in
    return img_cx + (tgt_cx - img_cx) * p, img_cy + (tgt_cy - img_cy) * p


frames_dir = f'frames/{video_name}'
os.makedirs(frames_dir, exist_ok=True)

half_w, half_h = half_w0, half_h0
for i in tqdm(range(frames)):
    cx, cy = center_for(half_w)
    frame = sample(cx, cy, half_w, half_h, resx, resy)
    if cmap is not None:
        plt.imsave(f'{frames_dir}/frame_{i:04d}.png', frame, cmap=cmap, vmin=0, vmax=1)
    else:
        plt.imsave(f'{frames_dir}/frame_{i:04d}.png', frame)
    half_w *= (1 - zoom_factor)
    half_h *= (1 - zoom_factor)

# High-res final frame.
cx, cy = center_for(half_w)
final = sample(cx, cy, half_w, half_h, final_resx, final_resy)
if cmap is not None:
    plt.imsave(f'{frames_dir}/FINAL.png', final, cmap=cmap, vmin=0, vmax=1)
else:
    plt.imsave(f'{frames_dir}/FINAL.png', final)

command = (f'ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%04d.png '
           f'-c:v libx264 -pix_fmt yuv420p -crf 20 -preset slow '
           f'{frames_dir}/{video_name}.mp4')
subprocess.run(command, shell=True, check=True)
print(f'Saved {frames_dir}/{video_name}.mp4')
