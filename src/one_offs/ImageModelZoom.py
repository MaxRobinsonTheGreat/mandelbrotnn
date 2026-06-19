"""Zoom into a trained model with the exact same motion as ImageZoom.py.

Companion to ImageZoom.py. Where the image zoom resolves into a finite grid of
crisp pixel blocks, this zoom queries the trained neural network instead -- the
same point, the same pan-and-zoom path -- revealing that the model is a smooth,
continuous function with no underlying pixel grid. Run the two side by side to
make the contrast obvious.

The model is the one defined and saved by train_image.py
(./models/<proj_name>.pt).
"""
import os
import subprocess
import sys
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import src.models as models
from src.videomaker import renderModel

# --- config (mirror ImageZoom.py) ----------------------------------------
model_name = 'hands_drawing_fourier'   # ./models/<model_name>.pt, from train_image.py
video_name = 'image_model_zoom_hands'
cmap = 'bone'

# Same point as ImageZoom.py: fraction of the image, (0,0)=top-left, (1,1)=bottom-right.
center = (0.31, 0.5)

resx, resy = 960, 832          # output frame resolution
frames = 400                   # number of frames
zoom_factor = 0.012             # fraction the half-window shrinks each frame (exponential)
final_resx, final_resy = 1920, 1664
fps = 60
max_gpu = True
# --------------------------------------------------------------------------

# Load the trained weights and rebuild the matching Fourier architecture. The
# shapes are inferred from the checkpoint so this works regardless of what the
# train_image.py __main__ block is currently set to.
state = torch.load(f'./models/{model_name}.pt')
hidden_size, in_features = state['inner_model.inLayer.weight'].shape
fourier_order = (in_features - 2) // 4
num_hidden_layers = sum(1 for k in state if k.startswith('inner_model.hidden.') and k.endswith('.weight'))
print(f'Loaded {model_name}: fourier_order={fourier_order}, '
      f'hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}')

model = models.Fourier(fourier_order, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
                       linmap=models.CenteredLinearMap(-1, 1, -1, 1, 2*torch.pi, 2*torch.pi)).cuda()
model.load_state_dict(state)
model.eval()

# The model lives in normalized coords x,y in [-1,1]. From ImageDataset's mapping
# (with its height flip), an image fraction (fx, fy) maps to:
#   mx = 2*fx - 1,  my = 1 - 2*fy
# The full image is the [-1,1]x[-1,1] window centered at the origin.
tgt_x = 2 * center[0] - 1
tgt_y = 1 - 2 * center[1]
img_x, img_y = 0.0, 0.0

# Half-window of the full view (the whole [-1,1] square).
half_w0, half_h0 = 1.0, 1.0


def center_for(half_w):
    """Pan from the image center to the target, tied to the zoom level so the
    target glides into place at a constant relative screen position."""
    p = 1 - half_w / half_w0  # 0 at full view, ->1 as we zoom all the way in
    return img_x + (tgt_x - img_x) * p, img_y + (tgt_y - img_y) * p


def render(cx, cy, half_w, half_h, out_w, out_h, full_gpu=max_gpu):
    # origin='lower' matches train_image.py's rendering orientation.
    # full_gpu=False batches by row -- needed for the high-res final frame, whose
    # point count is too large to fit the Fourier activations in one GPU batch.
    return renderModel(model, out_w, out_h,
                       xmin=cx - half_w, xmax=cx + half_w,
                       ymin=cy - half_h, ymax=cy + half_h,
                       max_gpu=full_gpu)


frames_dir = f'frames/{video_name}'
os.makedirs(frames_dir, exist_ok=True)

half_w, half_h = half_w0, half_h0
for i in tqdm(range(frames)):
    cx, cy = center_for(half_w)
    frame = render(cx, cy, half_w, half_h, resx, resy)
    plt.imsave(f'{frames_dir}/frame_{i:04d}.png', frame, cmap=cmap, vmin=0, vmax=1, origin='lower')
    half_w *= (1 - zoom_factor)
    half_h *= (1 - zoom_factor)

# High-res final frame.
cx, cy = center_for(half_w)
final = render(cx, cy, half_w, half_h, final_resx, final_resy, full_gpu=False)
plt.imsave(f'{frames_dir}/FINAL.png', final, cmap=cmap, vmin=0, vmax=1, origin='lower')

command = (f'ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%04d.png '
           f'-c:v libx264 -pix_fmt yuv420p -crf 20 -preset slow '
           f'{frames_dir}/{video_name}.mp4')
subprocess.run(command, shell=True, check=True)
print(f'Saved {frames_dir}/{video_name}.mp4')
