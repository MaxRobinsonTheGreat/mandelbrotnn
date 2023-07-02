from src.videomaker import renderMandelbrot, renderModel
import matplotlib.pyplot as plt
import os
import argparse
import sys
import subprocess
from tqdm import tqdm


# Define a function to parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Mandelbrot set zoom video.")
    parser.add_argument("resx", type=int, help="Width of the image.")
    parser.add_argument("resy", type=int, help="Height of the image.")
    parser.add_argument("frames", type=int, default=100, help="Number of frames")
    parser.add_argument("--xmin", type=float, default=-2.4, help="Minimum x value in the 2d space.")
    parser.add_argument("--xmax", type=float, default=1, help="Maximum x value in the 2d space.")
    parser.add_argument("--yoffset", type=float, default=0.5, help="Y offset.")
    parser.add_argument("--max_depth", type=int, default=500, help="Max depth param for mandelbrot function.")
    parser.add_argument("--zoom_speed", type=float, default=0.05, help="The fraction by which to zoom in each frame.")
    parser.add_argument("--video_name", type=str, default="mandelbrot", help="The name of the video.")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Create a directory for the frames if it doesn't exist
    frames_dir = f'frames/{args.video_name}'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    xmin, xmax = args.xmin, args.xmax
    yoffset = args.yoffset

    for i in tqdm(range(args.frames)):
        # Generate image and save it to a file
        image = renderMandelbrot(args.resx, args.resy, xmin=xmin, xmax=xmax, yoffset=yoffset, max_depth=args.max_depth, gpu=True)
        plt.imsave(f'{frames_dir}/frame_{i:03d}.png', image, vmin=0, vmax=1, cmap='gist_heat')

        # Update coordinates for zoom
        x_range = xmax - xmin
        xmin += args.zoom_speed * x_range / 2
        xmax -= args.zoom_speed * x_range / 2

    # Call FFmpeg to create the video
    video_name = f'{frames_dir}/{args.video_name}.mp4'
    command = f'ffmpeg -framerate 60 -i {frames_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 20 -preset slow {video_name}'
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    main()
    # python3 FractalZoom.py 3840 2160 6000 --xmin -1.86520494 --xmax 1.53479506 --yoffset 1.039049255 --max_depth 1500 --zoom_speed 0.0025 --video_name 4k_zoom
    # python3 FractalZoom.py 3840 2160 50 --xmin -1.86520494 --xmax 1.53479506 --yoffset 1.039049255 --max_depth 1500 --zoom_speed 0.01 --video_name test2