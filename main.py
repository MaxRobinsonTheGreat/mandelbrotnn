from src.videomaker import renderMandelbrot, renderModel, renderModelWindow, VideoMaker
from src.training import train
from src.hash_training import train as hash_train
from src.dataset import MandelbrotDataSet
from src import models
import matplotlib.pyplot as plt
import torch


def example_render():
    image = renderMandelbrot(7680, 4320, max_depth=1000, gpu=True).cpu()
    # image32 = image.type(torch.float32)
    # print(image[0][0])
    # plt.imsave('./captures/images/mandel_gpu_64.png', image.numpy(), vmin=0, vmax=1, cmap='gist_heat')
    plt.imsave('./captures/images/mandel_gpu_32.png', image, vmin=0, vmax=1, cmap='gist_heat')

    # import numpy as np

    # x = np.linspace(0, 1, image.shape[1])
    # y = np.linspace(0, 1, image.shape[0])
    # x, y = np.meshgrid(x, y)

    # # Create a figure
    # fig = plt.figure()

    # # Create a 3D axis
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the surface
    # ax.plot_surface(x, y, image, cmap='gray')

    # # Set the aspect ratio of the plot to match the aspect ratio of the image
    # ax.auto_scale_xyz([0, 1], [0, 1], [0, 1])

    # # Show the plot
    # plt.show()

    
    # plt.imshow(image, vmin=0, vmax=1, cmap='inferno')
    # plt.show()
    # 8k: 7680, 4320
    # 4k: 3840, 2160
    # 1080p: 1920, 1088
    # 960, 544
    # 480, 272

    # pass the following params to renderMandelbrot to zoom into useful locations:
    # xmin  xmax  yoffset
    # -1.8,  -0.9,  0.2,       leftmost bulb/tail
    # -0.9,  -0.1,  0.5,       left upper shoulder of main cardioid
    # -0.52,  0.29,  0.93,  Top anntenea
    # -0.18,  -0.13, 1.033, tiny mandelbrot


def example_train():
    # --- previous example (plain MLP/HashGrid via src.training.train) ---
    # print("Initializing model...")
    #
    # model = models.SkipConn(100, 10).cuda() # see src.models for more models
    # model = models.HashGrid(hidden_size=400, num_hidden_layers=50).cuda() # see src.models for more models
    
    # tiny HashGrid for a quick end-to-end test (~1.6M params, trains in seconds)
    
    # show the space before we've learned anything
    # plt.imshow(renderModel(model, 600, 600), vmin=0, vmax=1, cmap='inferno')
    # plt.show()
    
    dataset = MandelbrotDataSet(2000000, gpu=True) # generate a dataset with 200000 random training points
    eval_dataset = MandelbrotDataSet(100000, gpu=True)
    
    train(model, dataset, 1, batch_size=10000, eval_dataset=eval_dataset, oversample=0.1, use_scheduler=True, snapshots_every=50) # train for 20 epochs
    
    # show the space again
    # plt.imshow(renderModel(model, 600, 600), cmap='inferno')
    # plt.show()

def example_render_model():
    # saves a 4k image
    # model = models.Simple().cuda()
    true_mandelbrot = renderMandelbrot(3840, 2160, max_depth=1000, gpu=True)
    plt.imsave('./captures/images/true_mandelbrot_8k.png', true_mandelbrot, vmin=0, vmax=1, cmap='inferno')


    linmap = models.CenteredLinearMap(x_size=torch.pi*2, y_size=torch.pi*2)
    name = 'fourier_256_400_50'
    model = models.Fourier(fourier_order=256, hidden_size=400, num_hidden_layers=50, linmap=linmap)
    model.load_state_dict(torch.load('./models/'+name+'.pt'))
    model.cuda()
    image = renderModel(model, 3840, 2160, max_gpu=False, verbose=True)
    plt.imsave('./captures/images/'+name+'_8k.png', image, vmin=0, vmax=1, cmap='inferno')
    plt.show()


def example_train_capture():
    # we will caputre 480x480 video with new frame every 3 epochs
    shots = [
        {'frame':5, "xmin":-2.5, "xmax":1, "yoffset":0, "capture_rate":8},
        {'frame':10, "xmin":-1.8, "xmax":-0.9, "yoffset":0.2, "capture_rate":16},
    ]
    shots=None
    vidmaker = VideoMaker('mlp_basic', dims=(960, 544), capture_rate=3, shots=shots, max_gpu=True, cmap='inferno')
    # vidmaker = None
 
    # linmap = models.CenteredLinearMap(x_size=10, y_size=10)
    linmap = models.CenteredLinearMap(x_size=2, y_size=2)
    # linmap = models.CenteredLinearMap(x_size=torch.pi*2, y_size=torch.pi*2)
    # linmap = None

    # model = models.Simple(hidden_size=256, num_hidden_layers=6).cuda()
    model = models.SkipConn(hidden_size=256, num_hidden_layers=16, linmap=linmap)
    # model.load_state_dict(torch.load('./models/autosave.pt'))
    # model = models.Fourier(fourier_order=128, hidden_size=300, num_hidden_layers=30, linmap=linmap)
    # model = models.Fourier2D(12, 400, 50, linmap=linmap)
    # model = models.Taylor(10, 400, 50, linmap=linmap)
    # model.usePreprocessing()
    dataset = MandelbrotDataSet(1000000, max_depth=500, gpu=True, target='periodic')
    # dataset = MandelbrotDataSet(30000000, max_depth=1500, gpu=True)
    eval_dataset = MandelbrotDataSet(10000, max_depth=1500, gpu=True, target='periodic')
    # dataset = MandelbrotDataSet(loadfile='500k_inv')


    train(model, dataset, 40, batch_size=8000, 
        use_scheduler=True, oversample=0.1, eval_dataset=eval_dataset,
        snapshots_every=500, vm=vidmaker)


def train_champion(epochs=200, onfly_size=4_000_000, max_depth=200):
    """Train and render fractalsearch's best solution: 'hashgrid_errfield / n64l13'.

    The architecture (Instant-NGP hash grid with a fused Triton encoder: 13 levels,
    T=2^24, n_max=65536, 128x4 GELU decoder) and the training algorithm (dual learning
    rates 0.6/5e-3, warmup+cosine, persistent error-field mining with 98% hard samples,
    bf16) are exactly the champion's -- they're the defaults in hash_train. Fit to the
    periodic log-distance target over the canonical window, then rendered with our
    window-accurate renderers.

    Uses on-the-fly sampling (fractalsearch's ctx.sample): fresh points are generated and
    their target computed every step, so memory is constant and data is effectively
    unbounded -- no giant stored dataset. Runtime scales with `epochs`; `onfly_size` sets
    the notional dataset size (epoch length = onfly_size // batch_size).
    """
    print("Building champion HashGrid (Instant-NGP: 13 levels, T=2^24, n_max=65536, 128x4, Triton)...")
    model = models.HashGrid(n_levels=13, n_features=2, log2_hashmap_size=24,
                            n_min=16, n_max=65536, hidden_size=128, num_hidden_layers=4).cuda()

    # small fixed held-out set for a comparable metric (match its depth to the live target).
    # NOTE: fractalsearch's champion used max_depth=200; higher is sharper but costs more
    # per step since the on-the-fly target is recomputed live.
    eval_dataset = MandelbrotDataSet(200_000, max_depth=max_depth, gpu=True, target='periodic')

    # --- multi-shot zoom video: progressively dive onto the left-axis mini-bulb ---
    # Center (-1.755, 0); the depth-0.1 shot matches render_coords.txt. Shots fire at frame
    # thresholds, so spread them across the actual run length; the deepest holds to the end.
    # Deepest width ~0.03 is comfortably above the model's grid-resolution limit
    # (n_max=65536 over the 3.8-wide window); much past that shows interpolation.
    batch = 786_432
    capture_rate = 4
    cmap = 'inferno'
    total_steps = epochs * max(1, onfly_size // batch)
    frames = max(5, total_steps // capture_rate)
    windows = [(-2.65,  1.15),    # full set (context)
               (-2.095, -1.095),  # width 0.9, recentred on the bulb
               (-1.905, -1.605),  # width 0.3
               (-1.820, -1.720),  # width 0.1, 
               (-1.801, -1.769)]  # 
    # Evenly space the shots across the timeline regardless of how many windows there
    # are: window i fires at fraction i/len(windows), so the first is at 0.0 and the
    # last starts at (n-1)/n and holds to the end.
    n = len(windows)
    shots = [{'frame': int(i / n * frames), 'xmin': xm, 'xmax': xM, 'yoffset': 0.0}
             for i, (xm, xM) in enumerate(windows)]

    vm = VideoMaker('champion_2', dims=(960, 540), capture_rate=capture_rate, shots=shots,
                    max_gpu=True, cmap=cmap)

    # champion hyperparameters == hash_train defaults; only the big batch is set explicitly.
    # On-the-fly target: periodic log-distance, fp64, over the canonical window.
    # (If this OOMs, lower batch_size.)
    hash_train(model, None, epochs=epochs, batch_size=batch,
               on_the_fly=True, onfly_size=onfly_size, onfly_max_depth=max_depth,
               onfly_target='periodic', onfly_precision=64,
               eval_dataset=eval_dataset, savemodelas='champion.pt',
               snapshots_every=100, vm=vm, run_name='champion')

    # --- render the trained champion ---
    print("Rendering champion...")
    full = renderModelWindow(model, width=3840)          # full canonical window, square pixels
    plt.imsave('./captures/images/champion_full.png', full, vmin=0, vmax=1, cmap=cmap, origin='lower')

    # deep render at the saved coords (see render_coords.txt) for a model-vs-truth compare
    deep = renderModel(model, 3840, 2160, xmin=-1.805, xmax=-1.705,
                       ymin=-0.028125, ymax=0.028125, max_gpu=True)
    plt.imsave('./captures/images/champion_deep.png', deep, vmin=0, vmax=1, cmap=cmap, origin='lower')
    print("saved ./captures/images/champion_full.png and champion_deep.png")


def create_dataset():
    dataset = MandelbrotDataSet(100000, max_depth=50, gpu=True)
    dataset.save('1M_50_test')

if __name__ == "__main__":
    # create_dataset()
    # example_train()
    train_champion(5, 2_000_000)
    # example_render()
    # example_render_model()
    # example_train_capture()