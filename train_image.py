import torch
import src.models as models
import src.kan as kan
from src.imageDataset import ImageDataset
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
from src.videomaker import renderModel
from tqdm import tqdm
import os
import time

def trainImage(
        image_path,
        proj_name,
        model,
        lr,
        batch_size,
        num_epochs,
        optimizer='adam',
        scheduler_step=5,
        scheduler_gamma=0.5,
        save_every_n_batches=10,
        final_image_scale=-1, # if > 0, will save an extremely high quality image scaled up by this amount
        plt_color_map='inferno'
        ):
    
    # Create the dataset and data loader
    dataset = ImageDataset(image_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    resx, resy = dataset.width, dataset.height
    linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
    linspace = torch.rot90(linspace, 1, (0, 1))
    start_time = time.time()

    # Create the loss function and optimizer
    loss_func = nn.MSELoss()
    # loss_func = nn.L1Loss()
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


    # Train the model
    iteration, frame = 0, 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x, y = x.cuda(), y.cuda()

            # Forward pass
            y_pred = model(x).squeeze()

            # Compute loss
            loss = loss_func(y_pred, y)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            # Save an image of the model every n iterations
            if iteration % save_every_n_batches == 0:
                os.makedirs(f'./frames/{proj_name}', exist_ok=True)
                plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', renderModel(model, resx=resx, resy=resy, linspace=linspace, max_gpu=False), cmap=plt_color_map, origin='lower')
                frame += 1
            iteration += 1
            
        scheduler.step()

        # Log the average loss per epoch
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss / len(loader)}')

    # compute the loss of the final image
    final_image = renderModel(model, resx=resx, resy=resy, linspace=linspace, max_gpu=False, keep_cuda=True).cpu()
    final_loss = loss_func(final_image, dataset.image).item()
    with open(f'./frames/{proj_name}/stats.txt', 'w') as f:
        f.write(f'Final Loss: {final_loss}\n')
        f.write(f'Time taken: {time.time() - start_time:.2f} seconds\n')

    os.system(f'ffmpeg -y -r 30 -i ./frames/{proj_name}/%05d.png -c:v libx264 -preset veryslow -crf 18 ./frames/{proj_name}/{proj_name}.mp4')
    # ffmpeg -y -r 30 -i ./frames/mountain_SGD_fourier/%05d.png -c:v libx264 -preset veryslow -crf 18 ./frames/mountain_SGD_fourier/mountain_SGD_fourier.mp4


    # save an extremely high quality image
    if final_image_scale > 0:
        resx, resy = resx*final_image_scale, resy*final_image_scale
        linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
        linspace = torch.rot90(linspace, 1, (0, 1))
        plt.imsave(f'./frames/{proj_name}/{proj_name}_final.png', renderModel(model, resx=resx, resy=resy, linspace=linspace), cmap=plt_color_map, origin='lower')


if __name__ == '__main__':
    # model = models.Simple(hidden_size=200, num_hidden_layers=10, activation=nn.LeakyReLU).cuda()
    model = models.SkipConn(hidden_size=300, num_hidden_layers=15, activation=nn.GELU).cuda()
    # model = models.Fourier(16, hidden_size=200, num_hidden_layers=10, linmap=models.CenteredLinearMap(-1, 1, -1, 1, 2*torch.pi, 2*torch.pi)).cuda()
    trainImage(
        image_path='DatasetImages/biodiversity.png',
        proj_name='biodiversity_adam',
        model=model,
        lr=0.002,
        batch_size=8000,
        num_epochs=100,
        optimizer='adam',
        scheduler_step=10,
        scheduler_gamma=0.5,
        save_every_n_batches=20,
        plt_color_map='viridis',
    )
