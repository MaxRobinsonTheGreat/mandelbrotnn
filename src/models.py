import torch
import torch.nn as nn
import math

class Binary(nn.Module):
    def __init__(self):
        super(Binary, self).__init__()
    
    def forward(self, x):
        x[x>=0] = 1
        x[x<0] = -1
        return x

class Simple(nn.Module):
	""" 
	Very simple linear torch model. Uses relu activation and\
	one final sigmoid activation.

	Parameters: 
	hidden_size (float): number of parameters per hidden layer
	num_hidden_layers (float): number of hidden layers
	"""
	def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, activation=nn.GELU):
		super(Simple,self).__init__()
		layers = [nn.Linear(init_size, hidden_size),
							activation()]
		for _ in range(num_hidden_layers):
			layers.append(nn.Linear(hidden_size, hidden_size))
			layers.append(activation())
		layers.append(nn.Linear(hidden_size, 1))
		# layers.append(nn.Sigmoid())
		self.tanh = nn.Tanh()
		self.seq = nn.Sequential(*layers)

	def forward(self,x):
		return (self.tanh(self.seq(x))+1)/2


class SkipConn(nn.Module):
	""" 
	Linear torch model with skip connections between every hidden layer\
	as well as the original input appended to every layer.\
	Because of this, each hidden layer contains `2*hidden_size+init_size` params\
	due to skip connections.
	Uses relu activations and one final sigmoid activation.

	Parameters: 
	hidden_size (float): number of non-skip parameters per hidden layer
	num_hidden_layers (float): number of hidden layers
	"""
	def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, linmap=None, activation=nn.GELU):
		super(SkipConn,self).__init__()
		out_size = hidden_size

		self.inLayer = nn.Linear(init_size, out_size)
		self.activation = activation()
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
		cur = self.activation(self.inLayer(x))
		prev = torch.tensor([]).cuda()
		for layer in self.hidden:
			combined = torch.cat([cur, prev, x], 1)
			prev = cur
			cur = self.activation(layer(combined))
		y = self.outLayer(torch.cat([cur, prev, x], 1))
		return (self.tanh(y)+1)/2 # hey I think this works slightly better
		# return self.sig(y)



class Fourier(nn.Module):
	def __init__(self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None, activation=nn.GELU):
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
		self.inner_model = SkipConn(hidden_size, num_hidden_layers, fourier_order*4 + 2, activation=activation)
		# self.inner_model = Simple(hidden_size, num_hidden_layers, fourier_order*4 + 2, activation=activation)
		self._linmap = linmap
		self.orders = torch.arange(1, fourier_order + 1).float().to('cuda')

	def forward(self,x):
		if self._linmap:
			x = self._linmap.map(x)
		x = x.unsqueeze(-1)  # add an extra dimension for broadcasting
		fourier_features = torch.cat([torch.sin(self.orders * x), torch.cos(self.orders * x), x], dim=-1)
		fourier_features = fourier_features.view(x.shape[0], -1)  # flatten the last two dimensions
		return self.inner_model(fourier_features)


class Fourier2D(nn.Module):
    def __init__(self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None):
        super(Fourier2D,self).__init__()
        self.fourier_order = fourier_order
        self.inner_model = SkipConn(hidden_size, num_hidden_layers, (fourier_order*fourier_order*4) + 2)
        self._linmap = linmap
        self.orders = torch.arange(0, fourier_order).float().to('cuda')

    def forward(self,x):
        if self._linmap:
            x = self._linmap.map(x)
        features = [x]
        for n in self.orders:
            for m in self.orders:
                features.append((torch.cos(n*x[:,0])*torch.cos(m*x[:,1])).unsqueeze(-1))
                features.append((torch.cos(n*x[:,0])*torch.sin(m*x[:,1])).unsqueeze(-1))
                features.append((torch.sin(n*x[:,0])*torch.cos(m*x[:,1])).unsqueeze(-1))
                features.append((torch.sin(n*x[:,0])*torch.sin(m*x[:,1])).unsqueeze(-1))
        fourier_features = torch.cat(features, 1)
        return self.inner_model(fourier_features)


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
		x_b = -(xmin + xmax)*x_m/2
		y_b = -(ymin + ymax)*y_m/2
		self.m = torch.tensor([x_m, y_m], dtype=torch.float)
		self.b = torch.tensor([x_b, y_b], dtype=torch.float)


	def map(self, x):
		m = self.m.cuda()
		b = self.b.cuda()
		return m*x + b


_HASH_PRIMES = (1, 2654435761)


class HashGridEncoding(nn.Module):
	"""
	Multiresolution hash-grid encoding (Instant-NGP, Mueller et al. 2022).

	A stack of `n_levels` learnable feature grids at geometrically-spaced
	resolutions. Each input point is looked up in every grid via bilinear
	interpolation of its 4 surrounding cell corners; coarse levels index the
	table directly, fine levels collide through a spatial hash. The per-level
	features are concatenated into a `n_levels * n_features` vector that a small
	MLP decodes. Inputs must be in [0,1]^2.

	Parameters:
	n_levels (int): number of resolution levels
	n_features (int): feature channels stored per grid entry
	log2_hashmap_size (int): log2 of the max entries per level (table size cap)
	n_min (int): coarsest grid resolution
	n_max (int): finest grid resolution
	init_scale (float): grid entries are initialized uniform in [-init_scale, init_scale]
	"""
	def __init__(self, n_levels=12, n_features=2, log2_hashmap_size=24, n_min=16, n_max=32768,
				 init_scale=1e-4):
		super().__init__()
		self.n_levels = n_levels
		self.n_features = n_features
		self.T = 1 << log2_hashmap_size
		b = (n_max / n_min) ** (1.0 / (n_levels - 1))
		res = [int(round(n_min * b ** l)) for l in range(n_levels)]
		self._res = res                      # plain python ints -> no per-forward CUDA sync
		self.tables = nn.ParameterList()
		self.sizes = []
		self._direct = []                    # whether level indexes directly (no hash)
		for l in range(n_levels):
			N = res[l]
			size = min(N * N, self.T)
			self.sizes.append(size)
			self._direct.append(N * N <= size)
			t = nn.Parameter(torch.empty(size, n_features).uniform_(-init_scale, init_scale))
			self.tables.append(t)
		self.out_dim = n_levels * n_features

	def _idx(self, ix, iy, size, N, direct):
		# direct index when the grid fits in the table, else spatial hash
		if direct:
			return (iy * N + ix) % size
		h = (ix * _HASH_PRIMES[0]) ^ (iy * _HASH_PRIMES[1])
		return h % size

	def forward(self, x):
		# x in [0,1]^2
		feats = []
		for l in range(self.n_levels):
			N = self._res[l]
			size = self.sizes[l]
			direct = self._direct[l]
			xs = x * (N - 1)
			x0 = torch.floor(xs).long()
			xf = xs - x0.float()
			ix0, iy0 = x0[:, 0], x0[:, 1]
			ix1 = (ix0 + 1).clamp(max=N - 1)
			iy1 = (iy0 + 1).clamp(max=N - 1)
			ix0 = ix0.clamp(0, N - 1)
			iy0 = iy0.clamp(0, N - 1)
			fx = xf[:, 0:1]
			fy = xf[:, 1:2]
			tbl = self.tables[l]
			c00 = tbl[self._idx(ix0, iy0, size, N, direct)]
			c10 = tbl[self._idx(ix1, iy0, size, N, direct)]
			c01 = tbl[self._idx(ix0, iy1, size, N, direct)]
			c11 = tbl[self._idx(ix1, iy1, size, N, direct)]
			c0 = c00 * (1 - fx) + c10 * fx
			c1 = c01 * (1 - fx) + c11 * fx
			feats.append(c0 * (1 - fy) + c1 * fy)
		return torch.cat(feats, dim=1)


class HashGrid(nn.Module):
	"""
	Instant-NGP multiresolution hash-grid encoding + small MLP decoder.

	The strongest architecture found by the fractalsearch experiments: learnable
	feature grids at many scales carry the spatial detail while a tiny GELU MLP
	decodes them. A universal function approximator (learnable interpolant + MLP)
	that fits dense 2D fields with structure at every scale far better than a
	plain coordinate MLP.

	Inputs are raw (x, y) coordinates; they are normalized to [0,1]^2 internally
	using the [xmin,xmax]x[ymin,ymax] window (defaults to the standard Mandelbrot
	view) before the grid lookup.

	Parameters:
	n_levels (int): number of resolution levels in the hash grid
	n_features (int): feature channels per grid entry
	log2_hashmap_size (int): log2 of the per-level table size cap
	n_min (int): coarsest grid resolution
	n_max (int): finest grid resolution
	hidden_size (int): width of the decoder MLP
	num_hidden_layers (int): number of hidden layers in the decoder MLP
	in_size (int): input dimension size, default 2 (must be 2)
	out_size (int): output dimension size, default 1
	activation (nn.Module): decoder MLP activation, default nn.GELU
	init_scale (float): hash-grid entry init range, default 1e-4
	xmin/xmax/ymin/ymax (float): input window mapped to [0,1]^2
	"""
	def __init__(self, n_levels=12, n_features=2, log2_hashmap_size=24, n_min=16, n_max=32768,
				 hidden_size=128, num_hidden_layers=4, in_size=2, out_size=1, linmap=None,
				 activation=nn.GELU, init_scale=1e-4,
				 xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1):
		super().__init__()
		assert in_size == 2, "HashGrid is 2D-specific (bilinear interpolation over x, y)"
		self.enc = HashGridEncoding(n_levels, n_features, log2_hashmap_size, n_min, n_max, init_scale)
		dims = [self.enc.out_dim] + [hidden_size] * (num_hidden_layers - 1) + [out_size]
		blocks = []
		for i in range(len(dims) - 2):
			blocks += [nn.Linear(dims[i], dims[i + 1]), activation()]
		blocks += [nn.Linear(dims[-2], dims[-1])]
		self.net = nn.Sequential(*blocks)
		self.tanh = nn.Tanh()
		self._linmap = linmap
		self._cx, self._cy = xmin, ymin
		self._w, self._h = (xmax - xmin), (ymax - ymin)

	def _norm(self, x):
		nx = (x[:, 0:1] - self._cx) / self._w
		ny = (x[:, 1:2] - self._cy) / self._h
		return torch.cat([nx, ny], dim=1).clamp(0.0, 1.0)

	def forward(self, x):
		if self._linmap:
			x = self._linmap.map(x)
		e = self.enc(self._norm(x))
		return (self.tanh(self.net(e)) + 1) / 2


# Taylor features, x, x^2, x^3, ...
# surprisingly terrible
class Taylor(nn.Module):
	def __init__(self, taylor_order=4, hidden_size=100, num_hidden_layers=7, linmap=None):
		super(Taylor,self).__init__()
		self.taylor_order = taylor_order
		self._linmap = linmap
		self.inner_model = SkipConn(hidden_size, num_hidden_layers, taylor_order*2 + 2)

	def forward(self,x):
		if self._linmap:
			x = self._linmap.map(x)
		series = [x]
		for n in range(1, self.taylor_order+1):
			series.append(x**n)
		taylor = torch.cat(series, 1)
		return self.inner_model(taylor)

