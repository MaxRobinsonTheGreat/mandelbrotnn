import torch
import torch.nn as nn
import math

class Simple(nn.Module):
  """ 
  Very simple linear torch model. Uses relu activation and\
  one final sigmoid activation.

  Parameters: 
  hidden_size (float): number of parameters per hidden layer
  num_hidden_layers (float): number of hidden layers
  """
  def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2):
    super(Simple,self).__init__()
    layers = [nn.Linear(init_size, hidden_size),
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
  def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2):
    super(SkipConn,self).__init__()
    out_size = hidden_size

    self.inLayer = nn.Linear(init_size, out_size)
    self.relu = nn.ReLU()
    hidden = []
    for i in range(num_hidden_layers):
      in_size = out_size*2 + init_size if i>0 else out_size + init_size
      hidden.append(nn.Linear(in_size, out_size))
    self.hidden = nn.ModuleList(hidden)
    self.outLayer = nn.Linear(out_size*2+init_size, 1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    cur = self.inLayer(x)
    prev = torch.tensor([]).cuda()
    for layer in self.hidden:
      combined = torch.cat([prev, cur, x], 1)
      prev = cur
      cur = self.relu(layer(combined))
    y = self.outLayer(torch.cat([prev, cur, x], 1))
    return self.sig(y)


class Fourier(nn.Module):
  def __init__(self, fourier_order=4, inner_model=None):
    super(Fourier,self).__init__()
    self.fourier_order = fourier_order

    self.inner_model = inner_model
    if inner_model is None:
      self.inner_model = Simple(init_size=fourier_order*4 + 2)

  def forward(self,x):
    batch_size = x.shape[0]
    series = [x]
    for n in range(self.fourier_order):
      series.append(torch.cos(n*x*math.pi))
      series.append(torch.sin(n*x*math.pi))
    fourier = torch.cat(series, 1)
    return self.inner_model(fourier)
