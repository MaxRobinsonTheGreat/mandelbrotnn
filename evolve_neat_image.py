import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.problem.func_fit.func_fit import FuncFit
from tensorneat.common import ACT, AGG


def gaussian_(x):
    return jnp.exp(-x * x / 2.0)

ACT.add_func("gaussian", gaussian_)


class ImageFitProblem(FuncFit):
    jitable = True

    def __init__(self, image_path, grayscale=True, max_samples=None, error_method="mse"):
        image = Image.open(image_path)
        if grayscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')

        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array[::-1]

        height, width = img_array.shape[:2]
        self.img_height = height
        self.img_width = width
        self.is_grayscale = grayscale

        xs = np.linspace(-1, 1, width, dtype=np.float32)
        ys = np.linspace(-1, 1, height, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        if grayscale:
            targets = img_array.ravel()[:, None]
        else:
            targets = img_array.reshape(-1, 3)

        if max_samples is not None and max_samples < len(coords):
            indices = np.random.choice(len(coords), max_samples, replace=False)
            coords = coords[indices]
            targets = targets[indices]

        self._inputs = jnp.array(coords)
        self._targets = jnp.array(targets)

        print(f"Image: {width}x{height}, training on {len(coords)} samples, target mean={float(self._targets.mean()):.3f}")

        super().__init__(error_method=error_method)

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def input_shape(self):
        return self._inputs.shape

    @property
    def output_shape(self):
        return self._targets.shape

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        pass


def evolve_neat_image(
    image_path,
    pop_size=200,
    species_size=10,
    survival_threshold=0.2,
    init_hidden_layers=(4,),
    compatibility_threshold=2.0,
    max_stagnation=20,
    max_nodes=64,
    max_conns=256,
    generation_limit=500,
    max_samples=2048,
    activation_options=None,
    grayscale=True,
    seed=42,
):
    out_size = 1 if grayscale else 3

    if activation_options is None:
        activation_options = [ACT.tanh, ACT.sigmoid, ACT.sin, ACT.relu, ACT.identity, ACT.abs, ACT.gaussian]

    problem = ImageFitProblem(
        image_path=image_path,
        grayscale=grayscale,
        max_samples=max_samples,
    )

    algorithm = NEAT(
        pop_size=pop_size,
        species_size=species_size,
        survival_threshold=survival_threshold,
        compatibility_threshold=compatibility_threshold,
        max_stagnation=max_stagnation,
        genome=DefaultGenome(
            num_inputs=2,
            num_outputs=out_size,
            max_nodes=max_nodes,
            max_conns=max_conns,
            init_hidden_layers=init_hidden_layers,
            node_gene=BiasNode(
                activation_options=activation_options,
                aggregation_options=[AGG.sum, AGG.product],
            ),
            output_transform=ACT.sigmoid,
        ),
    )

    pipeline = Pipeline(
        algorithm=algorithm,
        problem=problem,
        seed=seed,
        generation_limit=generation_limit,
        fitness_target=0,
    )

    state = pipeline.setup()

    print("Compiling...")
    compiled_step = jax.jit(pipeline.step).lower(state).compile()
    print("Compilation done. Starting evolution...\n")

    start_time = time.time()
    gen_times = []

    for gen in range(generation_limit):
        pipeline.generation_timestamp = time.time()
        state, previous_pop, fitnesses = compiled_step(state)
        fitnesses_np = jax.device_get(fitnesses)
        pipeline.analysis(state, previous_pop, fitnesses_np)

        gen_time = time.time() - pipeline.generation_timestamp
        gen_times.append(gen_time)

        remaining = generation_limit - (gen + 1)
        avg_time = np.mean(gen_times[-20:])
        eta = remaining * avg_time
        eta_min, eta_sec = divmod(eta, 60)
        print(f"  -> Gen {gen+1}/{generation_limit} | Best fitness: {pipeline.best_fitness:.6f} | ETA: {int(eta_min)}m {int(eta_sec)}s\n")

        valid = fitnesses_np[~np.isinf(fitnesses_np)]
        if len(valid) > 0 and max(valid) >= 0:
            print("Perfect fitness reached!")
            break

    elapsed = time.time() - start_time
    print(f"\nDone. Total time: {elapsed:.1f}s | Best fitness: {pipeline.best_fitness:.6f}")

    if pipeline.best_genome is None:
        print("No valid genome found.")
        return

    print("Rendering final image at full resolution...")
    xs = np.linspace(-1, 1, problem.img_width, dtype=np.float32)
    ys = np.linspace(-1, 1, problem.img_height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    full_coords = jnp.array(np.stack([xx.ravel(), yy.ravel()], axis=-1))

    render_forward = jax.jit(jax.vmap(pipeline.algorithm.forward, in_axes=(None, None, 0)))
    render_transform = jax.jit(pipeline.algorithm.transform)

    transformed = render_transform(state, pipeline.best_genome)
    preds = render_forward(state, transformed, full_coords)
    preds = jnp.clip(preds, 0, 1)
    preds = jax.device_get(preds)

    if grayscale:
        result = preds.reshape(problem.img_height, problem.img_width)
    else:
        result = preds.reshape(problem.img_height, problem.img_width, 3)

    target_img = Image.open(image_path)
    if grayscale:
        target_img = target_img.convert('L')
    target_arr = np.array(target_img, dtype=np.float32) / 255.0
    target_arr = target_arr[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    cmap = 'gray' if grayscale else None
    axes[0].imshow(target_arr, cmap=cmap, origin='lower', vmin=0, vmax=1)
    axes[0].set_title("Target")
    axes[0].axis('off')
    axes[1].imshow(result, cmap=cmap, origin='lower', vmin=0, vmax=1)
    axes[1].set_title(f"NEAT result (fitness: {pipeline.best_fitness:.6f})")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig("neat_result.png", dpi=150)
    plt.show()


if __name__ == '__main__':
    evolve_neat_image(
        image_path='DatasetImages/blob_small.png',
        pop_size=100,
        species_size=10,
        survival_threshold=0.2,
        compatibility_threshold=1.5,
        max_stagnation=20,
        init_hidden_layers=(4,),
        generation_limit=300,
        max_samples=1024,
        max_nodes=32,
        max_conns=128,
        grayscale=True,
    )
