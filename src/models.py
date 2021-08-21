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
  Because of this, each hidden layer contains `2*hidden_size+2` params\
  due to skip connections.
  Uses relu activations and one final sigmoid activation.

  Parameters: 
  hidden_size (float): number of non-skip parameters per hidden layer
  num_hidden_layers (float): number of hidden layers
  """
  def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2):
    super(SkipConn,self).__init__()
    out_size = hidden_size

    self.inLayer = nn.Linear(init_size, out_size)
    self.relu = nn.LeakyReLU()
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
  def __init__(self, fourier_order=4, hidden_size=100, num_hidden_layers=7):
    """ 
    Linear torch model that adds Fourier Features to the initial input x as \
    sin(x) + cos(x), sin(2x) + cos(2x), sin(3x) + cos(3x), ...
    These features are then inputted to a SkipConn network.

    Parameters: 
    fourier_order (int): number fourier features to use. Each addition adds 4x\
     parameters to each layer.
    hidden_size (float): number of non-skip parameters per hidden layer (SkipConn)
    num_hidden_layers (float): number of hidden layers (SkipConn)
    """
    super(Fourier,self).__init__()
    self.fourier_order = fourier_order
    self.preprocess=False
    self.inner_model = SkipConn(hidden_size, num_hidden_layers, fourier_order*4 + 2)

  def usePreprocessing(self, xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1, period_size=1):
    """
    Call to use preprocessing on all model inputs. After calling, the forward function will\
    translate all input x,y coordinates into the range [-period_size/2, period_size/2]

    Parameters: 
    xmin (float): minimum x value in the 2d space
    xmax (float): maximum x value in the 2d space
    ymin (float): minimum y value in the 2d space
    ymax (float): maximum y value in the 2d space
    period_size (float): range of the space the values will be translated to [-period_size/2, period_size/2]
    """
    self.preprocess=True

    # very simple linear tranformation to apply to every value defined by f(x)=mx+b
    # must be done for x and y, and meant to "squeeze" x and y values into range (-pi/2, pi/2)
    x_m = period_size/(xmax - xmin)
    x_b = (xmin - xmax)/2
    y_m = period_size/(ymax - ymin)
    y_b = (ymin -ymax)/2
    self._temp_m = torch.tensor([[x_m, y_m]])
    self._temp_b = torch.tensor([[x_b, y_b]])

  def _preprocess(self, x):
    # private function used to preprocess. applies the mx+b translation to x
    if self._temp_m.shape[0] != x.shape[0]:
      batch_size = x.shape[0]
      self._temp_m = torch.stack([self._temp_m[0] for _ in range(batch_size)])
      self._temp_b = torch.stack([self._temp_b[0] for _ in range(batch_size)])
    m = self._temp_m.cuda()
    b = self._temp_b.cuda()

    return m*x + b

  def forward(self,x):
    if self.preprocess:
      x = self._preprocess(x)
    series = [x]
    for n in range(1, self.fourier_order+1):
      series.append(torch.sin(n*x*2*math.pi))
      series.append(torch.cos(n*x*2*math.pi))
    fourier = torch.cat(series, 1)
    return self.inner_model(fourier)


# Taylor features, x, x^2, x^3, ...
# surprisingly terrible
class Taylor(nn.Module):
  def __init__(self, taylor_order=4, hidden_size=100, num_hidden_layers=7):
    super(Taylor,self).__init__()
    self.taylor_order = taylor_order

    self.inner_model = SkipConn(hidden_size, num_hidden_layers, taylor_order*2 + 2)

  def forward(self,x):
    series = [x]
    for n in range(1, self.taylor_order+1):
      series.append(x**n)
    taylor = torch.cat(series, 1)
    return self.inner_model(taylor)

