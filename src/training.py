import torch
import models
from dataset import MandelbrotDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from videomaker import VideoMaker, modelGenerate

def train(model, dataset, epochs, batch_size=1000, use_scheduler=False, savemodelas='autosave.pt', vm=None):
    print("Initializing...")
    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-10)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)
    bne = torch.nn.BCELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print('Training...')
    avg_losses = []
    tot_iterations = 0
    for epoch in range(epochs):
        loop = tqdm(total=len(loader), position=0)
        tot_loss = 0

        for i, (inputs, outputs) in enumerate(loader):
            if vm is not None and tot_iterations%vm.capture_rate==0:
                vm.generateFrame(model)
            inputs, outputs = inputs.cuda(), outputs.cuda()

            optim.zero_grad()

            pred = model(inputs).squeeze()
            loss = bne(pred.float(), outputs.float())
            loss.backward()
            optim.step()
            tot_loss += loss.item()

            loop.set_description('epoch:{:d} Loss:{:.6f}'.format(epoch, tot_loss/(i+1)))
            loop.update(1)
            tot_iterations+=1
        loop.close()
        avg_losses.append(tot_loss/len(loader))

        if use_scheduler:
            scheduler.step()
        
        if savemodelas is not None:
            torch.save(model.state_dict(), './models/'+savemodelas)
    print("Finished training.")

    if vm is not None:
        print("Finalizing capture...")
        vm.generateFrame(model)
        vm.finish()
    if savemodelas is not None:
        print("Saving...")
        torch.save(model.state_dict(), './models/'+savemodelas)
    print("Done.")
    fig = plt.figure()
    fig.title('Avg Loss vs Epochs')
    plt.plot(avg_losses)
    plt.show()
