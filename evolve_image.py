import torch
import src.models as models
from src.imageDataset import ImageDataset
from torch import nn
import matplotlib.pyplot as plt
from src.videomaker import renderModel
from tqdm import tqdm
import os
import copy
import random
import time
import src.activations as activations

def evolveImage(image_path,
                proj_name,
                create_model,
                lr=1,
                lr_decay=0.9,
                reset_lr_at=-1,
                num_rounds=2000,
                pop_size=1000,
                num_species=5,
                init_rand_searches=1000,
                save_every_n_mutations=1,
                final_image_scale=-1,
                grayscale=True,
                plt_color_map='inferno',
                max_samples=None,
                ):

    dataset = ImageDataset(image_path, grayscale=grayscale)
    resx, resy = dataset.width, dataset.height
    tensor_image = dataset.image.squeeze().cuda()
    orig_lr = lr

    linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
    linspace = torch.rot90(linspace, 1, (0, 1))
    flat_linspace = linspace.reshape(-1, 2)
    flat_target = tensor_image.reshape(-1) if grayscale else tensor_image.reshape(3, -1).T
    if max_samples and max_samples >= flat_linspace.shape[0]:
        max_samples = None

    loss_func = nn.MSELoss()

    def sampleCoords(n):
        idx = torch.randint(0, flat_linspace.shape[0], (n,), device='cuda')
        return flat_linspace[idx], flat_target[idx]

    def evalLoss(net, coords=None, targets=None):
        if coords is None:
            # full image eval (for snapshots/final)
            im = renderModel(net, resx=resx, resy=resy, rgb=not grayscale, linspace=linspace, max_gpu=True, keep_cuda=True)
            if not grayscale:
                im = torch.permute(im, (2, 0, 1))
            return loss_func(im, tensor_image)
        out = net(coords).squeeze()
        return loss_func(out, targets)

    def generateImTensor(net):
        im = renderModel(net, resx=resx, resy=resy, rgb=not grayscale, linspace=linspace, max_gpu=True, keep_cuda=True)
        if not grayscale:
            im = torch.permute(im, (2, 0, 1))
        return im

    start_time = time.time()

    print(f'Random search for {init_rand_searches} rounds, keeping top {num_species} species...')
    candidates = []
    if max_samples:
        rs_coords, rs_targets = sampleCoords(max_samples)
    for _ in tqdm(range(init_rand_searches)):
        rand_net = create_model().cuda().eval()
        with torch.no_grad():
            loss = evalLoss(rand_net, rs_coords if max_samples else None, rs_targets if max_samples else None).item()
        candidates.append((loss, rand_net))

    candidates.sort(key=lambda x: x[0])
    species = [{'net': c[1], 'loss': c[0]} for c in candidates[:num_species]]
    del candidates

    best_species_idx = 0
    print(f'Species losses: {[f"{s["loss"]:.4f}" for s in species]}')

    def calc_diversity():
        total, n_pairs = 0.0, 0
        for i in range(len(species)):
            for j in range(i + 1, len(species)):
                for p1, p2 in zip(species[i]['net'].parameters(), species[j]['net'].parameters()):
                    total += torch.mean((p1.data - p2.data) ** 2).item()
                n_pairs += 1
        return total / n_pairs if n_pairs > 0 else 0.0

    print(f'Initial diversity: {calc_diversity():.4f}')

    mutation_names = ['mutate_some', 'mutate_one', 'crossover']
    mutation_wins = [0, 0, 0]

    frame = 0
    round_mutations = 0
    os.makedirs(f'./frames/{proj_name}', exist_ok=True)

    with torch.no_grad():
        initial_image = generateImTensor(species[best_species_idx]['net']).cpu().numpy()
    if not grayscale and initial_image.ndim == 3 and initial_image.shape[0] == 3:
        initial_image = initial_image.transpose(1, 2, 0)
    plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', initial_image, cmap=plt_color_map if grayscale else None, origin='lower')
    frame += 1

    for rnd in range(num_rounds):
        round_start_time = time.time()
        any_improvement = False

        if max_samples:
            rnd_coords, rnd_targets = sampleCoords(max_samples)

        total_evals = num_species * pop_size
        pbar = tqdm(total=total_evals, desc=f'Round {rnd}')
        for si, sp in enumerate(species):
            net = sp['net']
            sp_best_loss = sp['loss']
            sp_new_best = False
            sp_best_child = None
            sp_best_mutation = None

            for _ in range(pop_size):
                net_child = copy.deepcopy(net).eval()

                mutation_type = random.randint(0, 2)

                if mutation_type == 0:
                    params = list(net_child.parameters())
                    params = random.sample(params, random.randint(1, len(params)))
                    for param in params:
                        mutator = torch.randn_like(param.data) * lr
                        sparsity = random.uniform(0.01, 1)
                        mask = (torch.rand_like(mutator) < sparsity)
                        param.data += torch.where(mask, mutator, torch.zeros_like(mutator))
                elif mutation_type == 1:
                    params = list(net_child.parameters())
                    param = random.choice(params)
                    mutator = random.uniform(-1, 1) * lr
                    idx = tuple(torch.randint(0, s, (1,)).item() for s in param.data.shape)
                    param.data[idx] += mutator
                elif mutation_type == 2 and len(species) > 1:
                    other_idx = random.choice([j for j in range(len(species)) if j != si])
                    other_params = list(species[other_idx]['net'].parameters())
                    child_params = list(net_child.parameters())
                    num_to_swap = random.randint(1, len(child_params))
                    swap_indices = random.sample(range(len(child_params)), num_to_swap)
                    for idx in swap_indices:
                        child_params[idx].data.copy_(other_params[idx].data)

                with torch.no_grad():
                    loss = evalLoss(net_child, rnd_coords if max_samples else None, rnd_targets if max_samples else None).item()
                if loss < sp_best_loss:
                    sp_best_child = net_child
                    sp_best_loss = loss
                    sp_new_best = True
                    sp_best_mutation = mutation_type
                pbar.update(1)

            if sp_new_best:
                sp['net'] = sp_best_child
                sp['loss'] = sp_best_loss
                sp['winning_mutation'] = sp_best_mutation
                any_improvement = True
        pbar.close()

        prev_best_idx = best_species_idx
        best_species_idx = min(range(len(species)), key=lambda i: species[i]['loss'])
        if any_improvement and 'winning_mutation' in species[best_species_idx]:
            mutation_wins[species[best_species_idx]['winning_mutation']] += 1

        if any_improvement and round_mutations % save_every_n_mutations == 0:
            with torch.no_grad():
                best_im = generateImTensor(species[best_species_idx]['net']).cpu().numpy()
            if not grayscale and best_im.ndim == 3 and best_im.shape[0] == 3:
                best_im = best_im.transpose(1, 2, 0)
            plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', best_im, cmap=plt_color_map if grayscale else None, origin='lower')
            frame += 1
        round_mutations += 1

        if rnd == 0:
            time_taken = time.time() - round_start_time
            est_time = time_taken * num_rounds / 60 / 60
            print(f'Time taken: {time_taken:.2f} seconds | Estimated time: {est_time:.2f} hours')

        diversity = 0.0 #calc_diversity()

        wins = [f'{mutation_names[i]}: {mutation_wins[i]}' for i in range(3)]
        print(f'Round {rnd} | Best: S{best_species_idx} ({species[best_species_idx]["loss"]:.4f}) | Div: {diversity:.4f} | Lr: {lr:.4f} | {" | ".join(wins)}')

        lr *= lr_decay
        if lr < reset_lr_at and reset_lr_at > 0:
            lr = orig_lr

    best_net = species[best_species_idx]['net']
    final_image = generateImTensor(best_net)
    final_loss = loss_func(final_image, tensor_image).item()
    with open(f'./frames/{proj_name}/stats.txt', 'w') as f:
        f.write(f'Final Loss: {final_loss}\n')
        f.write(f'Time taken: {time.time() - start_time:.2f} seconds\n')
        f.write(f'Best species: {best_species_idx}\n')
        for i in range(3):
            f.write(f'{mutation_names[i]} wins: {mutation_wins[i]}\n')

    os.system(f'ffmpeg -y -r 30 -i ./frames/{proj_name}/%05d.png -c:v libx264 -preset veryslow -crf 18 ./frames/{proj_name}/{proj_name}.mp4')

    if final_image_scale > 0:
        resx, resy = resx*final_image_scale, resy*final_image_scale
        linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
        linspace = torch.rot90(linspace, 1, (0, 1))
        high_res_image = renderModel(best_net, resx=resx, resy=resy, linspace=linspace)
        if not grayscale and high_res_image.ndim == 3 and high_res_image.shape[0] == 3:
            high_res_image = high_res_image.transpose(1, 2, 0)
        plt.imsave(f'./frames/{proj_name}/{proj_name}_final.png', high_res_image, cmap=plt_color_map if grayscale else None, origin='lower')


class SinActivation(nn.Module): 
    def __init__(self):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)

if __name__ == '__main__':
    # create_model = lambda: models.Simple(hidden_size=50, num_hidden_layers=5, activation=activations.MexicanHatActivation)
    # create_model = lambda: models.SkipConn(hidden_size=50, num_hidden_layers=5, activation=activations.MorletActivation)
    # create_model = lambda: models.Fourier(16, hidden_size=50, num_hidden_layers=5, linmap=models.CenteredLinearMap(-1, 1, -1, 1, 2*torch.pi, 2*torch.pi))
    
    # Example for grayscale image
    # evolveImage(
    #     image_path='DatasetImages/darwin_grey.jpg',
    #     proj_name='test2',
    #     create_model=lambda: models.SkipConn(hidden_size=50, num_hidden_layers=5, activation=activations.MorletActivation),
    #     lr=1,
    #     lr_decay=0.9,
    #     reset_lr_at=0.01,
    #     num_rounds=20,
    #     pop_size=1000,
    #     init_rand_searches=1,
    #     save_every_n_mutations=1,
    #     grayscale=True
    # )
    
    # Example for RGB image
    evolveImage(
        image_path='DatasetImages/smiley_small.png',
        proj_name='test2',
        create_model=lambda: models.SkipConn(out_size=1, hidden_size=10, num_hidden_layers=2, activation=nn.GELU),
        lr=1,
        lr_decay=0.9,
        reset_lr_at=0.01,
        num_rounds=100,
        pop_size=2000,
        num_species=1,
        init_rand_searches=1000,
        save_every_n_mutations=1,
        grayscale=True,
        plt_color_map='inferno',
        max_samples=10000,
    )
