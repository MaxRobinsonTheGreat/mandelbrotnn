import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from src.videomaker import renderMandelbrot
ar = 16/9
# candidate mini-mandelbrot framings along the far-left antenna (y=0)
cands = [
    ("p3_wide",   -1.7548,  0.0, 0.060, 2000),
    ("p3_mid",    -1.7548,  0.0, 0.020, 3000),
    ("x1786",     -1.7864,  0.0, 0.030, 2500),
    ("p4_1941",   -1.9408,  0.0, 0.030, 3000),
    ("x1968",     -1.9683,  0.0, 0.020, 3000),
    ("x1749",     -1.7490,  0.0, 0.020, 3000),
    ("x1770",     -1.7700,  0.0, 0.030, 2500),
]
for name, cx, cy, w, md in cands:
    h=w/ar; xmin,xmax,ymin,ymax=cx-w/2,cx+w/2,cy-h/2,cy+h/2
    im=torch.tensor(renderMandelbrot(480,270,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,max_depth=md,gpu=True,precision=64))
    print(f"{name:10s} cx={cx} w={w:.3f} std={im.std():.3f} mean={im.mean():.3f} inset={(im>=0.999).float().mean():.2f}")
