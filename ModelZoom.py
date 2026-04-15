import torch
from src.videomaker import renderMandelbrot, renderModel
import matplotlib.pyplot as plt
import os
import argparse
import sys
import subprocess
from tqdm import tqdm
import src.models as models

resx = 960
resy = 544
# resx = 480
# resy = 272
frames = 2000
xmin = -2.417156607
xmax = -0.417156607
# xmin = -2.5
# xmax = 1.0
yoffset = 0
zoom_speed = 0.0025

final_resx = 1920
final_resy = 1088

model_name = 'fourier_256_400_50'
video_name = 'zoom_in_' + model_name
max_gpu = True

#load model
linmap = models.CenteredLinearMap(x_size=torch.pi*2, y_size=torch.pi*2)
model = models.Fourier(fourier_order=256, hidden_size=400, num_hidden_layers=50, linmap=linmap).cuda()
model.load_state_dict(torch.load('./models/'+model_name+'.pt')) # you need to have a model with this name


frames_dir = f'frames/{video_name}'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

xmin, xmax = xmin, xmax
yoffset = yoffset

for i in tqdm(range(frames)):
    # Generate image and save it to a file
    image = renderModel(model, resx, resy, xmin=xmin, xmax=xmax, yoffset=yoffset, max_gpu=max_gpu)
    plt.imsave(f'{frames_dir}/frame_{i:03d}.png', image, vmin=0, vmax=1, cmap='gist_heat')

    # Update coordinates for zoom
    x_range = xmax - xmin
    xmin += zoom_speed * x_range / 2
    xmax -= zoom_speed * x_range / 2

image = renderModel(model, final_resx, final_resy, xmin=xmin, xmax=xmax, yoffset=yoffset, max_gpu=False)
plt.imsave(f'{frames_dir}/FINAL.png', image, vmin=0, vmax=1, cmap='gist_heat')

# Call FFmpeg to create the video
video_name = f'{frames_dir}/{video_name}.mp4'
command = f'ffmpeg -framerate 60 -i {frames_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 20 -preset slow {video_name}'
subprocess.run(command, shell=True, check=True)