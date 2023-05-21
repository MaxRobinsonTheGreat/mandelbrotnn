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
	def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, linmap=None):
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
		self.tanh = nn.Tanh()
		self.sig = nn.Sigmoid()
		self._linmap = linmap

	def forward(self, x):
		if self._linmap:
			x = self._linmap.map(x)
		cur = self.relu(self.inLayer(x))
		prev = torch.tensor([]).cuda()
		for layer in self.hidden:
			combined = torch.cat([cur, prev, x], 1)
			prev = cur
			cur = self.relu(layer(combined))
		y = self.outLayer(torch.cat([cur, prev, x], 1))
		# return (self.tanh(y)+1)/2 # hey I think this works slightly better
		return self.sig(y)


# class SkipConn2(nn.Module):
# 	def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2):
# 		super(SkipConn2,self).__init__()
# 		out_size = hidden_size
# 		self.features_len = 2+2+2+2
# 		self.inLayer = nn.Linear(self.features_len, out_size)
# 		self.relu = nn.LeakyReLU()
# 		hidden = []
# 		for i in range(num_hidden_layers):
# 			in_size = out_size*2 + self.features_len if i>0 else out_size + self.features_len
# 			hidden.append(nn.Linear(in_size, out_size))
# 			hidden.append(nn.Linear(out_size, out_size))
# 		self.hidden = nn.ModuleList(hidden)
# 		self.outLayer = nn.Linear(out_size*2+self.features_len, 1)
# 		self.sig = nn.Sigmoid()

# 	def forward(self, x):
# 		features = [x]
# 		features.append(torch.sin(x**2))
# 		features.append(torch.sin(x))
# 		features.append(torch.cos(x))
# 		features = torch.cat(features, 1)
# 		cur = self.relu(self.inLayer(features))
# 		prev = torch.tensor([]).cuda()
# 		for i in range(0,len(self.hidden), 2):
# 			layer1 = self.hidden[i]
# 			layer2 = self.hidden[i+1]
# 			combined = torch.cat([cur, prev, features], 1)
# 			prev = cur
# 			cur = self.relu(layer1(combined))
# 			cur = self.relu(layer2(cur))
# 		y = self.outLayer(torch.cat([cur, prev, features], 1))
# 		return self.sig(y)


class Fourier(nn.Module):
	def __init__(self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None):
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
		self.inner_model = SkipConn(hidden_size, num_hidden_layers, fourier_order*4 + 2)
		self._linmap = linmap

	def forward(self,x):
		if self._linmap:
			x = self._linmap.map(x)
		series = [x]
		for n in range(1, self.fourier_order+1):
			series.append(torch.sin(n*x))
			series.append(torch.cos(n*x))
		fourier = torch.cat(series, 1)
		return self.inner_model(fourier)


class CenteredLinearMap():
	def __init__(self, xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1, x_size=None, y_size=None):
		if x_size is not None:
			x_m = x_size/(xmax - xmin)
		else: 
			x_m = 1.
		if y_size is not None:
			y_m = y_size/(ymax - ymin)
		else: 
			y_m = 1.
		x_b = -(xmin + xmax)*x_m/2 - 1 # TODO REMOVE!
		y_b = -(ymin + ymax)*y_m/2
		self.m = torch.tensor([x_m, y_m], dtype=torch.float)
		self.b = torch.tensor([x_b, y_b], dtype=torch.float)
		print(self.m, self.b)


	def map(self, x):
		m = self.m.cuda()
		b = self.b.cuda()
		return m*x + b


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

