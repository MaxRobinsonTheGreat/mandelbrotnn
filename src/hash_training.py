"""Instant-NGP hash-grid training algorithm, wired into the mandelbrotnn harness.

This ports the winning fractalsearch training loop (fused-Triton multiresolution
hash grid + small MLP, "hashgrid_errfield / n64l13", MSE 0.000226 on the 5-minute
benchmark) into this project. The novel pieces of the algorithm are:

- **Dual learning rates.** The grid feature tables (``model.enc``) and the MLP
  decoder (``model.net``) get separate param groups. The tables are a big sparse
  lookup -- each entry is touched by only a few samples per step -- so they need a
  much larger LR (~0.6) than the dense MLP (~5e-3).
- **Warmup + cosine schedule.** LR ramps linearly for the first ``warmup_frac`` of
  training, then cosine-anneals down to ``lr_min_frac`` of peak.
- **Persistent error-field mining.** A coarse 2D grid over the input window keeps
  an EMA of the per-cell MEAN |error|, updated for free each step from the train
  batch's own residuals (mean, not sum, so oversampled cells don't self-reinforce).
  ``hard_frac`` of every batch is drawn by cell-multinomial over the field +
  uniform jitter inside the cell; the rest is uniform. This replaced pool-based
  hard mining: there are NO extra model forwards or target evaluations per step,
  so nearly all compute goes into optimization steps, and the temporally-averaged
  error signal is less noisy than any single-pool estimate.
- **bf16 autocast** for throughput (fp32 exponent range, so no GradScaler needed).

The public interface mirrors :func:`src.training.train` as closely as possible
(same argument names/order, TensorBoard logging, per-epoch saving, snapshot images,
and ``vm`` video capture) so it is a drop-in replacement. ``epochs`` is interpreted
exactly as in ``train``: the loop runs ``epochs * (len(dataset) // batch_size)``
optimization steps, which also sets the horizon for the LR schedule.

With ``on_the_fly=True`` the hard coords are sampled anywhere in the window and
their targets computed live (the champion's regime). With a fixed dataset, the same
field drives per-point importance sampling: every stored point is weighted by its
cell's field value, so ``dataset`` must expose ``.inputs`` (N, 2) and ``.outputs``
(N,) -- as ``MandelbrotDataSet`` does.
"""

import os, sys, math, datetime
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from src.videomaker import renderModel, renderModelWindow
from src.dataset import mandelbrotTensor, XMIN, XMAX, YMIN, YMAX
from logger import Logger

os.makedirs("./models", exist_ok=True)


def train(model, dataset, epochs, batch_size=1000, use_scheduler=True, oversample=0,
          eval_dataset=None, savemodelas='autosave.pt', snapshots_every=-1, vm=None,
          tbl_lr=0.6, mlp_lr=5e-3, hard_frac=0.98, field_width=2048, field_ema=0.6,
          warmup_frac=0.08, lr_min_frac=0.01, use_amp=True, grad_clip=None,
          run_name=None, on_the_fly=False, onfly_size=4_000_000, onfly_window=None,
          onfly_max_depth=200, onfly_target='periodic', onfly_precision=64):
    """
    Trains a hash-grid model with the Instant-NGP algorithm ported from fractalsearch.

    Mirrors :func:`src.training.train`; the parameters below it are hash-specific.

    Parameters:
    model (torch.nn.Module): model with 2 inputs and 1 output. For dual-LR training
        it should expose ``.enc`` (grid tables) and ``.net`` (decoder MLP), as
        ``models.HashGrid`` does; otherwise a single param group at ``mlp_lr`` is used.
    dataset (Dataset): must expose ``.inputs`` (N, 2) and ``.outputs`` (N,) tensors.
    epochs (int): number of epochs; total steps = epochs * (len(dataset)//batch_size).
    batch_size (int): optimization batch size.
    use_scheduler (bool): apply the warmup+cosine LR schedule. If False, LR is held
        at peak for the whole run.
    oversample (float): accepted for interface compatibility with ``train`` but
        ignored -- error-proportional mining (``hard_frac``) supersedes it.
    eval_dataset (Dataset): optional held-out set scored with MAE during training.
    savemodelas (str): filename under ./models to save to (per epoch). None disables.
    snapshots_every (int): log a rendered preview + eval every N steps (-1 disables).
    vm (VideoMaker): if given, captures a frame every ``vm.capture_rate`` steps and
        renders an mp4 at the end.
    tbl_lr (float): peak LR for the grid feature tables (``model.enc``).
    mlp_lr (float): peak LR for the decoder MLP (``model.net``).
    hard_frac (float): fraction of each batch drawn from the error field; the rest
        is uniform. The field tolerates extreme values (0.98 was best) because its
        EMA + mean-update provide implicit coverage.
    field_width (int): error-field cells along x; cells along y follow the window
        aspect ratio (2048 -> cells ~2 render pixels at 4K, the tuned optimum;
        finer fields lose to sparse/stale per-cell statistics).
    field_ema (float): per-update EMA retention of a visited cell's old value.
    warmup_frac (float): fraction of training spent linearly warming up the LR.
    lr_min_frac (float): floor of the cosine decay, as a fraction of peak LR.
    use_amp (bool): use bf16 autocast on CUDA.
    grad_clip (float): max grad norm; None or <=0 disables clipping (off by default,
        matching the original solution).
    run_name (str): optional label for the TensorBoard run. The run dir is prefixed
        with a sortable ``YYYYMMDD-HHMMSS`` timestamp so the newest run sorts last.
    on_the_fly (bool): if True, ignore ``dataset`` (may be None) and sample fresh random
        points + compute their target every step (fractalsearch's ``ctx.sample``). Gives
        unbounded data at constant memory, at the cost of recomputing targets each step.
    onfly_size (int): notional dataset size used only to set epoch length / the LR-schedule
        horizon when ``on_the_fly`` (steps_per_epoch = onfly_size // batch_size). Does not
        affect memory.
    onfly_window (tuple|None): (xmin, xmax, ymin, ymax) to sample over; defaults to the
        canonical window. Should match the model's normalization window.
    onfly_max_depth (int), onfly_target (str), onfly_precision (int): how the on-the-fly
        targets are computed (passed to mandelbrotTensor). Match these to ``eval_dataset``.
    """
    print("Initializing...")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", stamp if run_name is None else f"{stamp}_{run_name}")
    tb = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard run: {log_dir}")
    logger = Logger(__file__, dir=tb.log_dir)
    logger.copyFile(sys.argv[0])
    logger.createDir('images')
    logger.createDir('models')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dual param groups: grid tables learn fast, MLP learns slow. Fall back to a
    # single group for models that don't split into enc/net.
    if hasattr(model, "enc") and hasattr(model, "net"):
        groups = [
            {"params": model.enc.parameters(), "peak_lr": tbl_lr, "lr": tbl_lr},
            {"params": model.net.parameters(), "peak_lr": mlp_lr, "lr": mlp_lr},
        ]
    else:
        print("Model has no .enc/.net split; using a single param group at mlp_lr.")
        groups = [{"params": model.parameters(), "peak_lr": mlp_lr, "lr": mlp_lr}]
    optim = torch.optim.Adam(groups, betas=(0.9, 0.99), eps=1e-15)

    # The sampling window: on-the-fly draws fresh coords here; the error field always
    # spans it (fixed datasets are assumed to live inside the canonical window).
    xmin, xmax, ymin, ymax = onfly_window if onfly_window is not None else (XMIN, XMAX, YMIN, YMAX)

    # Data source: fresh on-the-fly samples, or a fixed dataset pinned on device.
    if on_the_fly:
        def compute_targets(coords):
            return mandelbrotTensor(coords[:, 1], coords[:, 0], onfly_max_depth,
                                    target=onfly_target,
                                    precision=onfly_precision).reshape(-1).to(torch.float32)

        N = onfly_size   # notional size -> epoch length / LR-schedule horizon only
        print(f"On-the-fly sampling: window=({xmin},{xmax},{ymin},{ymax}) max_depth={onfly_max_depth} "
              f"target={onfly_target} precision={onfly_precision} notional_size={N}")
    else:
        # field-driven importance sampling needs random access -> keep on device
        coords_all = dataset.inputs.to(device)
        targets_all = dataset.outputs.to(device).reshape(-1)
        N = coords_all.shape[0]

    steps_per_epoch = max(1, N // batch_size)
    total_iters = max(1, epochs * steps_per_epoch)
    n_hard = max(0, min(batch_size, int(round(hard_frac * batch_size))))
    n_unif = batch_size - n_hard

    # --- persistent error field (champion mining) --------------------------------
    # Coarse grid of EMA per-cell mean |error| over the window; starts uniform.
    FW = int(field_width)
    FH = max(1, int(round(FW * (ymax - ymin) / (xmax - xmin))))
    field = torch.ones(FH * FW, device=device)
    print(f"Error field: {FW}x{FH} cells, ema={field_ema}, hard_frac={hard_frac}")

    def field_cells(coords):
        ix = ((coords[:, 0] - xmin) / (xmax - xmin) * FW).long().clamp_(0, FW - 1)
        iy = ((coords[:, 1] - ymin) / (ymax - ymin) * FH).long().clamp_(0, FH - 1)
        return iy * FW + ix

    if not on_the_fly:
        cells_all = field_cells(coords_all)   # precomputed once for importance sampling

    def set_lr(frac):
        if frac < warmup_frac:
            mult = frac / warmup_frac
        else:
            f2 = (frac - warmup_frac) / max(1e-9, (1 - warmup_frac))
            mult = lr_min_frac + 0.5 * (1 - lr_min_frac) * (1 + math.cos(math.pi * f2))
        for g in optim.param_groups:
            g["lr"] = g["peak_lr"] * mult

    # bf16 autocast: fp32 exponent range so no GradScaler, grid gather/interp stay safe.
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_amp and device.type == "cuda") \
        else torch.autocast(device_type="cpu", enabled=False)

    print('Training...')
    model.train()
    tot_iterations = 0
    if eval_dataset is not None:
        tb.add_scalar('Loss/eval', evaluate(model, eval_dataset, batch_size), tot_iterations)

    for epoch in range(epochs):
        loop = tqdm(total=steps_per_epoch, position=0)
        tot_loss = 0
        for i in range(steps_per_epoch):
            if vm is not None and tot_iterations % vm.capture_rate == 0:
                vm.generateFrame(model)
            if use_scheduler:
                set_lr(min(1.0, tot_iterations / total_iters))

            # --- error-field mining: no pool, no extra forwards -------------------
            if on_the_fly:
                if n_hard > 0:
                    cell = torch.multinomial(field, n_hard, replacement=True)
                    u = torch.rand(n_hard, 2, device=device)
                    hx = ((cell % FW).float() + u[:, 0]) / FW * (xmax - xmin) + xmin
                    hy = (torch.div(cell, FW, rounding_mode='floor').float() + u[:, 1]) \
                        / FH * (ymax - ymin) + ymin
                    hard_c = torch.stack([hx, hy], dim=1)
                else:
                    hard_c = torch.empty(0, 2, device=device)
                ux = torch.rand(n_unif, device=device) * (xmax - xmin) + xmin
                uy = torch.rand(n_unif, device=device) * (ymax - ymin) + ymin
                coords = torch.cat([hard_c, torch.stack([ux, uy], dim=1)])
                targets = compute_targets(coords)
            else:
                # importance-sample stored points by their cell's field value
                if n_hard > 0:
                    hard = torch.multinomial(field[cells_all], n_hard, replacement=True)
                else:
                    hard = torch.empty(0, dtype=torch.long, device=device)
                unif = torch.randint(0, N, (n_unif,), device=device)
                idx = torch.cat([hard, unif])
                coords, targets = coords_all[idx], targets_all[idx]

            # --- optimize -------------------------------------------------------
            with amp:
                pred = model(coords).reshape(-1)
                loss = torch.mean((pred.float() - targets) ** 2)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            # --- free field update from this batch's residuals --------------------
            # Per-cell MEAN error (not sum) so oversampled cells don't self-reinforce.
            with torch.no_grad():
                err = (pred.detach().float() - targets.float()).abs()
                c = field_cells(coords)
                esum = torch.zeros_like(field).scatter_add_(0, c, err)
                cnt = torch.zeros_like(field).scatter_add_(0, c, torch.ones_like(err))
                seen = cnt > 0
                field[seen] = field_ema * field[seen] + (1 - field_ema) * (esum[seen] / cnt[seen])
                field.clamp_min_(1e-8)

            tot_loss += loss.item()
            loop.set_description('epoch:{:d} Loss:{:.6f}'.format(epoch, tot_loss / (i + 1)))
            loop.update(1)
            tb.add_scalar('Loss/train', loss.detach().item(), tot_iterations)
            tb.add_scalar('Learning Rate', optim.param_groups[0]['lr'], tot_iterations)
            tot_iterations += 1

            if snapshots_every != -1 and tot_iterations % snapshots_every == 0:
                tb.add_image('sample', renderModelWindow(model, width=960),
                             tot_iterations, dataformats='HW')
                if eval_dataset is not None:
                    tb.add_scalar('Loss/eval', evaluate(model, eval_dataset, batch_size), tot_iterations)
        loop.close()

        if savemodelas is not None:
            torch.save(model.state_dict(), './models/' + savemodelas)

    print("Finished training.")
    print("Final learning rate:", optim.param_groups[0]['lr'])
    if eval_dataset is not None:
        tb.add_scalar('Loss/eval', evaluate(model, eval_dataset, batch_size), tot_iterations)

    if vm is not None:
        print("Finalizing capture...")
        vm.generateFrame(model)
        vm.generateVideo()
    if savemodelas is not None:
        print("Saving...")
        torch.save(model.state_dict(), './models/' + savemodelas)
    print("Done.")
    plt.show()
    tb.close()


def evaluate(model, eval_dataset, batch_size):
    model.eval()
    with torch.no_grad():
        loader = DataLoader(eval_dataset, batch_size=batch_size)
        tot_loss = 0
        for i, (inputs, outputs, indices) in enumerate(loader):
            inputs, outputs = inputs.cuda(), outputs.cuda()
            pred = model(inputs).squeeze()
            pred, outputs = pred.float(), outputs.float()
            loss = torch.mean(torch.abs(outputs - pred))
            tot_loss += loss.item()
    model.train()
    return tot_loss / len(loader)
