import torch
import src.models as models
from src.imageDataset import ImageDataset
from torch import optim, nn
import matplotlib.pyplot as plt
from src.videomaker import renderModel
from tqdm import tqdm
import os
import copy
import random

# Define variables
image_path = 'DatasetImages/smiley.png'
hidden_size = 100
num_hidden_layers = 10
lr = 1
lr_decay = 0.99
num_rounds = 10
pop_size = 10000
proj_name = 'evolve_smiley_skipconn'
save_every_n_mutations = 1


# Create the dataset
dataset = ImageDataset(image_path)
resx, resy = dataset.width, dataset.height
tensor_image = dataset.image.squeeze().cuda().rot90(2)

linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(-1, 1, resy)), dim=-1).cuda()

# Create the model
net = models.SkipConn(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers).cuda()
# linmap = models.CenteredLinearMap(-1, 1, -1, 1, 2*torch.pi, 2*torch.pi)
# net = models.Fourier(2, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, linmap=linmap).cuda()


# Create the loss function (anti-fitness function)
loss_func = nn.MSELoss()

# print(renderModel(net, resx=resx, resy=resy, linspace=linspace, max_gpu=True, keep_cuda=True).shape, tensor_image.shape)

def generateImTensor(nn):
    return renderModel(nn, resx=resx, resy=resy, linspace=linspace, max_gpu=True, keep_cuda=True)

best_loss = loss_func(generateImTensor(net), tensor_image)
frame = 0
mutations = 0
for round in range(num_rounds):


    new_best = False
    best = net

    for child in tqdm(range(pop_size)):
        net_child = copy.deepcopy(net)
        params = list(net_child.parameters())
        params = random.sample(params, random.randint(1, len(params)))
        for param in params:

            mutator = torch.randn_like(param.data) * lr
            filter_prob = random.uniform(0, 1)
            mutator = torch.where(torch.rand_like(mutator) < filter_prob, mutator, torch.zeros_like(mutator))
            param.data = param.data + mutator
        net_child.eval()


        loss = loss_func(generateImTensor(net_child), tensor_image)
        if loss < best_loss:
            best = net_child
            best_loss = loss
            new_best = True
            if mutations % save_every_n_mutations == 0:
                os.makedirs(f'./frames/{proj_name}', exist_ok=True)
                plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', renderModel(best, resx=resx, resy=resy, linspace=linspace, max_gpu=True), cmap='viridis', origin='lower')
                frame += 1
                mutations += 1

    if new_best:
        # redraw the approximated function
        net = best

    print(f'Round: {round} | Current Loss: {best_loss.item():.4f} | Lr: {lr:.4f}')    
    lr *= lr_decay

os.system(f'ffmpeg -y -r 30 -i ./frames/{proj_name}/%05d.png -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p ./frames/{proj_name}/{proj_name}.mp4')
# save an extremely high quality image
resx, resy = resx*4, resy*4
linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(-1, 1, resy)), dim=-1).cuda()
plt.imsave(f'./frames/{proj_name}/{proj_name}_final.png', renderModel(net, resx=resx, resy=resy, linspace=linspace), cmap='viridis', origin='lower')
