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

def evolveImage(image_path,
                proj_name,
                create_model, # function that returns a model so we can rerandomize it
                lr=1,
                lr_decay=0.9,
                reset_lr_at=-1,
                num_rounds=2000,
                pop_size=1000,
                init_rand_searches=1000,
                save_every_n_mutations=1,
                final_image_scale=-1, # if > 0, will save an extremely high quality image scaled up by this amount
                plt_color_map='inferno'
                ):

    # Create the dataset
    dataset = ImageDataset(image_path)
    resx, resy = dataset.width, dataset.height
    tensor_image = dataset.image.squeeze().cuda()
    orig_lr = lr

    linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
    linspace = torch.rot90(linspace, 1, (0, 1))

    net = create_model().cuda()

    # loss function (anti-fitness function)
    loss_func = nn.MSELoss()
    # loss_func = nn.L1Loss()

    def generateImTensor(nn):
        return renderModel(nn, resx=resx, resy=resy, linspace=linspace, max_gpu=True, keep_cuda=True)
    
    start_time = time.time()

    best_loss = loss_func(generateImTensor(net), tensor_image)
    print(f'Initial loss: {best_loss.item():.4f}')
    print(f'Random search for {init_rand_searches} rounds...')
    for _ in tqdm(range(init_rand_searches)):
        rand_net = create_model().cuda()
        rand_net.eval()
        loss = loss_func(generateImTensor(rand_net), tensor_image)
        if loss < best_loss:
            best_loss = loss
            net = rand_net

    print(f'Best loss: {best_loss.item():.4f}')
    frame = 0
    mutations = 0
    os.makedirs(f'./frames/{proj_name}', exist_ok=True)
    plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', generateImTensor(net).cpu().numpy(), cmap=plt_color_map, origin='lower')
    frame += 1
    for round in range(num_rounds):
        new_best = False
        best = net
        best_mutation_type = None
        best_im = None

        round_start_time = time.time()
        for child in tqdm(range(pop_size)):
            net_child = copy.deepcopy(net)

            mutation_type = random.randint(0, 1)
            if mutation_type == 0: # mutate some
                params = list(net_child.parameters())
                params = random.sample(params, random.randint(1, len(params)))
                for param in params:
                    # Add a small constant to prevent mutations from becoming too small
                    mutator = torch.randn_like(param.data) * lr
                    sparsity = random.uniform(0.01, 1)
                    mask = (torch.rand_like(mutator) < sparsity)
                    param.data += torch.where(mask, mutator, torch.zeros_like(mutator))
            elif mutation_type == 1: # mutate one
                params = list(net_child.parameters())
                params = random.choice(params)
                mutator = random.uniform(-1, 1) * lr
                idx = tuple(torch.randint(0, s, (1,)).item() for s in params.data.shape)
                params.data[idx] += mutator

            # clip params
            # for param in net_child.parameters():
            #     param.data = torch.clamp(param.data, -1, 1)

            net_child.eval()
            # plt.imsave(f'./frames/{proj_name}/{child}.png', renderModel(net_child, resx=resx, resy=resy, linspace=linspace, max_gpu=True), cmap=plt_color_map, origin='lower')

            im = generateImTensor(net_child)
            loss = loss_func(im, tensor_image)
            if loss < best_loss:
                best = net_child
                best_loss = loss
                new_best = True
                best_mutation_type = mutation_type
                best_im = im

        if new_best:
            net = best
            print(f'Mutated {"one" if best_mutation_type == 1 else "many"} params')
            if mutations % save_every_n_mutations == 0:
                
                plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', best_im.cpu().numpy(), cmap=plt_color_map, origin='lower')
                frame += 1
                mutations += 1
        
        if round == 0:
            time_taken = time.time() - round_start_time 
            est_time = time_taken * num_rounds / 60 / 60
            print(f'Time taken: {time_taken:.2f} seconds | Estimated time: {est_time:.2f} hours')


        print(f'Round: {round} | Current Loss: {best_loss.item():.4f} | Lr: {lr:.4f}')    
        lr *= lr_decay
        if lr < reset_lr_at and reset_lr_at > 0:
            lr = orig_lr

    final_image = generateImTensor(net)
    final_loss = loss_func(final_image, tensor_image).item()
    with open(f'./frames/{proj_name}/stats.txt', 'w') as f:
        f.write(f'Final Loss: {final_loss}\n')
        f.write(f'Time taken: {time.time() - start_time:.2f} seconds\n')

    # render
    os.system(f'ffmpeg -y -r 30 -i ./frames/{proj_name}/%05d.png -c:v libx264 -preset veryslow -crf 18 ./frames/{proj_name}/{proj_name}.mp4')


    
    if final_image_scale > 0:
        resx, resy = resx*final_image_scale, resy*final_image_scale
        linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
        linspace = torch.rot90(linspace, 1, (0, 1))
        plt.imsave(f'./frames/{proj_name}/{proj_name}_final.png', renderModel(net, resx=resx, resy=resy, linspace=linspace), cmap=plt_color_map, origin='lower')


if __name__ == '__main__':
    create_model = lambda: models.Simple(hidden_size=50, num_hidden_layers=5, activation=models.Binary)
    # create_model = lambda: models.SkipConn(hidden_size=50, num_hidden_layers=5, activation=models.Binary)
    # create_model = lambda: models.Fourier(16, hidden_size=50, num_hidden_layers=5, linmap=models.CenteredLinearMap(-1, 1, -1, 1, 2*torch.pi, 2*torch.pi))
    evolveImage(
        image_path='DatasetImages/darwin_grey.jpg',
        proj_name='test',
        create_model=create_model,
        lr=1,
        lr_decay=0.9,
        reset_lr_at=0.01,
        num_rounds=10,
        pop_size=1000,
        init_rand_searches=1,
        save_every_n_mutations=1
    )
