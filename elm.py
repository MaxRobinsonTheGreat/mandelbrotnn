import torch
import torch.nn as nn
from src.videomaker import renderModel
from src.dataset import MandelbrotDataSet
from src.imageDataset import ImageDataset
import matplotlib.pyplot as plt
import numpy as np

class ELM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ELM, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        self.activation = nn.SELU()

        # Initialize hidden layer with random weights and turn off gradients
        nn.init.uniform_(self.hidden_layer.weight)
        self.hidden_layer.weight.requires_grad = False

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Create some random data
# dataset = MandelbrotDataSet(7000, max_depth=1500, gpu=True)
# x = dataset.inputs.float()
# y = torch.unsqueeze(dataset.outputs, 1).float()


dataset = ImageDataset('./DatasetImages/blob_small.png')
# set x to the input values of the dataset
x = torch.stack([dataset[i][0] for i in range(len(dataset))])
# set y to the output values of the dataset
y = torch.unsqueeze(torch.stack([dataset[i][1] for i in range(len(dataset))]), 1)
print(x.shape, y.shape)

# Create the model
model = ELM(2, 1000, 1)

def evaluate_model(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        mse = ((outputs - y) ** 2).mean().item()
    return mse

print('Before training, MSE: {:f}'.format(evaluate_model(model, x, y)))

# Forward pass to get the output of the hidden layer
H = model.hidden_layer(x)
H = model.activation(H)

# Calculate the output weights
print(H.shape)
H_pinv = torch.linalg.pinv(H)
print(H_pinv.shape, y.shape)
output_weights = torch.mm(H_pinv, y)

# output_weights = np.dot(pinv2(hidden_nodes(X_train)), y_train)

# print(model.output_layer.weight.data.size(), output_weights.size())
# Set the output weights
model.output_layer.weight.data = output_weights.view(model.output_layer.weight.data.size())

print('After training, MSE: {:f}'.format(evaluate_model(model, x, y)))

model.cuda()
resx, resy = dataset.width, dataset.height
linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
#rotate the linspace 90 degrees
linspace = torch.rot90(linspace, 1, (0, 1))
plt.imshow(renderModel(model, resx=resx, resy=resy, linspace=linspace), cmap='magma', origin='lower')
plt.show()