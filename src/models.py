import torch
import torch.nn as nn


class Simple(nn.Module):
  """ 
  Very simple linear torch model. Uses relu activation and\
  one final sigmoid activation.

  Parameters: 
  hidden_size (float): number of parameters per hidden layer
  num_hidden_layers (float): number of hidden layers
  """
  def __init__(self, hidden_size=100, num_hidden_layers=7):
    super(Simple,self).__init__()
    layers = [nn.Linear(2, hidden_size),
              nn.ReLU()]
    for _ in range(num_hidden_layers):
      layers.append(nn.Linear(hidden_size, hidden_size))
      layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, 1))
    layers.append(nn.Sigmoid())

    self.seq = nn.Sequential(*layers)

  def forward(self,x):
    return self.seq(x)


class SkipConn(nn.Module):
  """ 
  Linear torch model with skip connections between every hidden layer\
  as well as the original input appended to every layer.\
  Because of this, each hidden layer contains 2*hidden_size +2 params\
  due to skip connections.
  Uses relu activations and one final sigmoid activation.

  Parameters: 
  hidden_size (float): number of novel parameters per hidden layer\
  (not including skip connections)
  num_hidden_layers (float): number of hidden layers
  """
  def __init__(self, hidden_size=100, num_hidden_layers=7):
    super(SkipConn,self).__init__()
    out_size = hidden_size

    self.inLayer = nn.Linear(2, out_size)
    self.relu = nn.ReLU()
    hidden = []
    for i in range(num_hidden_layers):
      in_size = out_size*2 + 2 if i>0 else out_size + 2
      hidden.append(nn.Linear(in_size, out_size))
    self.hidden = nn.ModuleList(hidden)
    self.outLayer = nn.Linear(out_size+2, 1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    cur = self.inLayer(x)
    prev = torch.tensor([]).cuda()
    for layer in self.hidden:
      combined = torch.cat([prev, cur, x], 1)
      prev = cur
      cur = self.relu(layer(combined))
    y = self.outLayer(torch.cat([cur, x], 1))
    return self.sig(y)
