import torch
import torch.nn as nn
import triton
import triton.language as tl

# model sourcecode is optimized via fractalsearch

_HASH_PRIME_Y = tl.constexpr(2654435761)


@triton.jit
def _hashgrid_fwd(x_ptr, tbl_ptr, out_ptr, meta_ptr, B,
				  L: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
	# meta layout per level: [N, size, offset, direct]  (int64 x 4)
	pid = tl.program_id(0)
	o = pid * BLOCK + tl.arange(0, BLOCK)
	m = o < B
	x = tl.load(x_ptr + o * 2, mask=m, other=0.0).to(tl.float32)
	y = tl.load(x_ptr + o * 2 + 1, mask=m, other=0.0).to(tl.float32)
	for l in tl.static_range(L):
		N = tl.load(meta_ptr + l * 4 + 0)
		size = tl.load(meta_ptr + l * 4 + 1)
		off = tl.load(meta_ptr + l * 4 + 2)
		direct = tl.load(meta_ptr + l * 4 + 3)
		xs = x * (N - 1).to(tl.float32)
		ys = y * (N - 1).to(tl.float32)
		ix0 = xs.to(tl.int64)            # x,y in [0,1] -> floor == trunc
		iy0 = ys.to(tl.int64)
		fx = xs - ix0.to(tl.float32)
		fy = ys - iy0.to(tl.float32)
		ix1 = tl.minimum(ix0 + 1, N - 1)
		iy1 = tl.minimum(iy0 + 1, N - 1)
		for f in tl.static_range(F):
			acc = tl.zeros((BLOCK,), dtype=tl.float32)
			# corners: (00),(10),(01),(11)
			for c in tl.static_range(4):
				ix = ix0 if (c % 2 == 0) else ix1
				iy = iy0 if (c // 2 == 0) else iy1
				wx = (1.0 - fx) if (c % 2 == 0) else fx
				wy = (1.0 - fy) if (c // 2 == 0) else fy
				idx_d = iy * N + ix
				idx_h = ((ix * 1) ^ (iy * _HASH_PRIME_Y)) % size
				idx = tl.where(direct != 0, idx_d, idx_h) + off
				v = tl.load(tbl_ptr + idx * F + f, mask=m, other=0.0)
				acc += v * wx * wy
			tl.store(out_ptr + o * (L * F) + l * F + f, acc, mask=m)


@triton.jit
def _hashgrid_bwd(x_ptr, go_ptr, gtbl_ptr, meta_ptr, B,
				  L: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
	pid = tl.program_id(0)
	o = pid * BLOCK + tl.arange(0, BLOCK)
	m = o < B
	x = tl.load(x_ptr + o * 2, mask=m, other=0.0).to(tl.float32)
	y = tl.load(x_ptr + o * 2 + 1, mask=m, other=0.0).to(tl.float32)
	for l in tl.static_range(L):
		N = tl.load(meta_ptr + l * 4 + 0)
		size = tl.load(meta_ptr + l * 4 + 1)
		off = tl.load(meta_ptr + l * 4 + 2)
		direct = tl.load(meta_ptr + l * 4 + 3)
		xs = x * (N - 1).to(tl.float32)
		ys = y * (N - 1).to(tl.float32)
		ix0 = xs.to(tl.int64)
		iy0 = ys.to(tl.int64)
		fx = xs - ix0.to(tl.float32)
		fy = ys - iy0.to(tl.float32)
		ix1 = tl.minimum(ix0 + 1, N - 1)
		iy1 = tl.minimum(iy0 + 1, N - 1)
		for f in tl.static_range(F):
			g = tl.load(go_ptr + o * (L * F) + l * F + f, mask=m, other=0.0)
			for c in tl.static_range(4):
				ix = ix0 if (c % 2 == 0) else ix1
				iy = iy0 if (c // 2 == 0) else iy1
				wx = (1.0 - fx) if (c % 2 == 0) else fx
				wy = (1.0 - fy) if (c // 2 == 0) else fy
				idx_d = iy * N + ix
				idx_h = ((ix * 1) ^ (iy * _HASH_PRIME_Y)) % size
				idx = tl.where(direct != 0, idx_d, idx_h) + off
				tl.atomic_add(gtbl_ptr + idx * F + f, g * wx * wy, mask=m)


class _HashGridFn(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, table, meta, L, F):
		x = x.contiguous().float()
		B = x.shape[0]
		out = torch.empty(B, L * F, device=x.device, dtype=torch.float32)
		grid = (triton.cdiv(B, 256),)
		_hashgrid_fwd[grid](x, table, out, meta, B, L=L, F=F, BLOCK=256)
		ctx.save_for_backward(x, meta)
		ctx.shape = (table.shape, L, F)
		return out

	@staticmethod
	def backward(ctx, go):
		x, meta = ctx.saved_tensors
		tshape, L, F = ctx.shape
		B = x.shape[0]
		gtbl = torch.zeros(tshape, device=x.device, dtype=torch.float32)
		grid = (triton.cdiv(B, 256),)
		_hashgrid_bwd[grid](x, go.contiguous().float(), gtbl, meta, B, L=L, F=F, BLOCK=256)
		return None, gtbl, None, None, None


class HashGridEncoding(nn.Module):
	"""
	Multiresolution hash-grid encoding (Instant-NGP, Mueller et al. 2022), computed
	by a fused Triton kernel (fractalsearch champion encoder).

	A stack of `n_levels` learnable feature grids at geometrically-spaced
	resolutions, packed into ONE contiguous table. Each input point is looked up in
	every grid via bilinear interpolation of its 4 surrounding cell corners; coarse
	levels index the table directly, fine levels collide through a spatial hash.
	Index computation + gather + interpolation run in a single kernel pass with no
	intermediate tensors (~2x faster than the per-level torch loop it replaces);
	the backward recomputes indices and atomic-adds into the table gradient.
	Inputs must be in [0,1]^2. CUDA only.

	Parameters:
	n_levels (int): number of resolution levels
	n_features (int): feature channels stored per grid entry
	log2_hashmap_size (int): log2 of the max entries per level (table size cap)
	n_min (int): coarsest grid resolution
	n_max (int): finest grid resolution
	init_scale (float): grid entries are initialized uniform in [-init_scale, init_scale]
	"""
	def __init__(self, n_levels=13, n_features=2, log2_hashmap_size=24, n_min=16, n_max=65536,
				 init_scale=1e-4):
		super().__init__()
		self.n_levels = n_levels
		self.n_features = n_features
		self.T = 1 << log2_hashmap_size
		b = (n_max / n_min) ** (1.0 / (n_levels - 1))
		res = [int(round(n_min * b ** l)) for l in range(n_levels)]
		sizes = [min(N * N, self.T) for N in res]
		offsets = [0]
		for s in sizes:
			offsets.append(offsets[-1] + s)
		meta = []
		for l in range(n_levels):
			meta += [res[l], sizes[l], offsets[l], 1 if res[l] * res[l] <= sizes[l] else 0]
		self.register_buffer("meta", torch.tensor(meta, dtype=torch.long))
		self.table = nn.Parameter(
			torch.empty(offsets[-1], n_features).uniform_(-init_scale, init_scale))
		self.out_dim = n_levels * n_features

	def forward(self, x):
		# x in [0,1]^2 -> (B, n_levels * n_features), fp32
		return _HashGridFn.apply(x, self.table, self.meta, self.n_levels, self.n_features)


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
	def __init__(self, n_levels=13, n_features=2, log2_hashmap_size=24, n_min=16, n_max=65536,
				 hidden_size=128, num_hidden_layers=4, in_size=2, out_size=1, linmap=None,
				 activation=nn.GELU, init_scale=1e-4,
				 xmin=-2.65, xmax=1.15, ymin=-1.2, ymax=1.2):
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
