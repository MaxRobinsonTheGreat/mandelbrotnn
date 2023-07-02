import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from src.videomaker import renderModel
import os, sys

from torch.utils.tensorboard import SummaryWriter
from logger import Logger

os.makedirs("./models", exist_ok=True)


def train(model, dataset, epochs, batch_size=1000, use_scheduler=False, oversample=0, eval_dataset=None, savemodelas='autosave.pt', snapshots_every=-1, vm=None):
    """ 
    Trains the given model on the given dataset for the given number of epochs. Can save the model and
    capture training videos as it goes. 

    Parameters: 
    model (torch.nn.Module): torch model with 2 inputs and 1 output. Will mutate this object.
    dataset (torch.utils.data.Dataset): torch dataset
    epochs (int): Number of epochs to train for
    batch_size (int): batch size for dataloader
    use_scheduler (bool): whether or not to use the simple StepLR scheduler.\
        Defaults to False.
    savemodelas (string): name of the file to save the model to. If None, the\
        model will not be saved. The model is automatically saved every epoch\
        to allow for interruption. Defaults to "autosave.mp4".
    vm (VideoMaker): used to capture training images and save them as a mp4.\
        If None, will not save a video capture (this will increase perfomance).\
        Defaults to None.
    """
    print("Initializing...")
    tb = SummaryWriter()
    logger = Logger(__file__, dir=tb.log_dir)
    logger.copyFile(sys.argv[0])

    logger.createDir('images')
    logger.createDir('models')

    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
        # I have experimented with other supposedly better schedulers before, they don't work as well
        # try this one if you'd like:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=20)
    # bne = torch.nn.BCELoss()
    
    if oversample != 0:
        per_batch = math.floor(batch_size * oversample)
        dataset.start_oversample(math.floor(len(dataset) * oversample))

    print('Training...')
    avg_losses = []
    tot_iterations = 0
    if eval_dataset is not None:
        tb.add_scalar('Loss/eval', evaluate(model, eval_dataset, batch_size), tot_iterations)

    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loop = tqdm(total=len(loader), position=0)
        tot_loss = 0

        for i, (inputs, outputs, indices) in enumerate(loader):
            if vm is not None and tot_iterations%vm.capture_rate==0:
                vm.generateFrame(model)
            inputs, outputs = inputs.cuda(), outputs.cuda()

            optim.zero_grad()

            pred = model(inputs).squeeze()
            pred, outputs = pred.float(), outputs.float()
            
            # all_losses = (outputs - pred)**2
            # all_losses = torch.sqrt(torch.abs(pred - outputs))
            all_losses = torch.abs(outputs - pred)
            # all_losses = loss_fn(pred, outputs)

            if oversample != 0:
                size = per_batch if per_batch < len(all_losses) else len(all_losses)
                highest_loss = torch.topk(all_losses, size)
                selected_indices = highest_loss[1].cpu()
                dataset.add_oversample(indices[selected_indices])

            # loss = bne(pred.float(), outputs.float())
            # loss = torch.mean(torch.abs(pred - outputs))
            loss = torch.mean(all_losses)
            # loss = torch.mean(2 * (1-(1/(torch.abs(outputs - pred)+1))))
            # loss = torch.mean(torch.sqrt(torch.abs(outputs**2 - pred**2)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optim.step()
            tot_loss += loss.item()

            loop.set_description('epoch:{:d} Loss:{:.6f}'.format(epoch, tot_loss/(i+1)))
            loop.update(1)
            tb.add_scalar('Loss/train', loss.detach().item(), tot_iterations)
            tb.add_scalar('Learning Rate', scheduler.get_last_lr()[0], tot_iterations)
            tot_iterations+=1
            inputs, outputs = inputs.cpu(), outputs.cpu()
            torch.cuda.empty_cache()
            # scheduler.step()

            if snapshots_every != -1 and tot_iterations%snapshots_every == 0:
                tb.add_image('sample', renderModel(model, 960, 544, max_gpu=True).data, tot_iterations)
                if eval_dataset is not None:
                    tb.add_scalar('Loss/eval', evaluate(model, eval_dataset, batch_size), tot_iterations)
        loop.close()
        avg_losses.append(tot_loss/len(loader))

        if use_scheduler:
            scheduler.step()
        dataset.update_oversample()
        
        if savemodelas is not None:
            torch.save(model.state_dict(), './models/'+savemodelas)
    print("Finished training.")
    print("Final learning rate:", scheduler.get_last_lr()[0])
    if eval_dataset is not None:
        tb.add_scalar('Loss/eval', evaluate(model, eval_dataset, batch_size), tot_iterations)

    if vm is not None:
        print("Finalizing capture...")
        vm.generateFrame(model)
        vm.generateVideo()
    if savemodelas is not None:
        print("Saving...")
        torch.save(model.state_dict(), './models/'+savemodelas)
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
    return tot_loss/len(loader)