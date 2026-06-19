import math
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

device="cuda"

# Canonical view window -------------------------------------------------------
# Single source of truth for the x/y input ranges, matching the fractalsearch
# project. The dataset and the renderers (src/videomaker.py) default to this.
XMIN, XMAX = -2.65, 1.15
YMIN, YMAX = -1.2, 1.2

# Target definitions ----------------------------------------------------------
# Two ways to turn escape-time iteration into a [0,1] value, selectable via the
# `target` argument throughout this module:
#   'smooth'   - the project's legacy monotonic squash (smoothMandelbrot). Saturates
#                toward 1.0 with more iterations.
#   'periodic' - periodic log-distance target ported from fractalsearch. Tracks the
#                orbit derivative for a distance-to-set estimate, then folds it through
#                a sine so detail never saturates and fine structure is exposed at every
#                scale. Harder to fit; great for high-capacity models like HashGrid.
#                NOTE: it bails out at a large escape radius, so it wants a higher
#                max_depth than 'smooth' (e.g. 200) to resolve exterior points.
PERIODIC_ESCAPE_R = 1.0e4   # escape radius / bailout for the distance estimate
PERIODIC_BETA = 0.050       # periodic frequency: phase = BETA * log(distance)
_TWO_PI = 2.0 * math.pi

# function helper, don't directly call
def _m(a, max_depth, target='smooth'):
	if target == 'periodic':
		return _m_periodic(a, max_depth)
	z = 0
	for n in range(max_depth):
		z = z**2 + a
		if abs(z) > 2:
			# return 0
			# return min((n-1)/max_depth, 1)
			# return -(math.cos((n-1)*math.pi/50)/2) + (1/2)
			# return math.log(n/10 + 1)
			return smoothMandelbrot(n)
	return 1.0

# function helper, don't directly call. Scalar periodic log-distance target.
def _m_periodic(c, max_depth):
	z = 0j
	dz = 0j
	for n in range(max_depth):
		dz = 2 * z * dz + 1          # z'_{n+1} = 2*z_n*z'_n + 1 (uses z before update)
		z = z * z + c
		if abs(z) > PERIODIC_ESCAPE_R:
			zmag = abs(z)
			dzmag = max(abs(dz), 1e-20)
			dist = zmag * math.log(max(zmag, 1.0001)) / dzmag
			phase = PERIODIC_BETA * math.log(max(dist, 1e-30))
			return 0.5 + 0.5 * math.sin(_TWO_PI * phase)
	return 1.0

def smoothMandelbrot(iters, smoothness=50):
	return 1-(1/((iters/smoothness) + 1))

def mandelbrot(x, y, max_depth=50, target='smooth', precision=64):
	"""
	Calculates whether the given point is in the mandelbrot set.

	Parameters:
	x (float): real part of the number
	y (float): complex part of the number
	max_depth (int): Maximum number of recursive steps before deciding\
	whether the value is in the mandelbrot set
	target (str): 'smooth' (legacy monotonic squash) or 'periodic' (periodic\
	log-distance target). See the module-level notes.
	precision (int): accepted for API parity with the GPU paths; the scalar CPU\
	computation always runs in Python double precision, so this is a no-op here.


	Returns:
	float: Number between 1 and 0 where 1.0 is in the mandelbrot set and\
	values closer to 1.0 required more steps to determine this
	"""
	return _m(x + 1j * y, max_depth, target=target)

def _dtypes(precision):
	"""Map a precision (64 or 32) to (real dtype, complex dtype)."""
	if precision == 32:
		return torch.float32, torch.complex64
	return torch.float64, torch.complex128

def mandelbrotGPU(resx, resy, xmin, xmax, ymin, ymax, max_depth, target='smooth', precision=64):
	real_dtype, _ = _dtypes(precision)
	X = torch.linspace(xmin, xmax, resx, device=device, dtype=real_dtype)
	Y = torch.linspace(ymin, ymax, resy, device=device, dtype=real_dtype)

	# Create the meshgrid using real and imaginary ranges
	imag_values, real_values = torch.meshgrid(Y, X)

	return mandelbrotTensor(imag_values, real_values, max_depth, target=target, precision=precision)

def _mandelbrot_periodic_tensor(c, max_depth):
	# Periodic log-distance target (ported from fractalsearch), vectorized over `c`.
	z = torch.zeros_like(c)
	dz = torch.zeros_like(c)              # orbit derivative z' = dz/dc (z'_0 = 0)
	z_esc = torch.zeros_like(c)           # z and z' captured at the escape iteration
	dz_esc = torch.zeros_like(c)
	alive = torch.ones(c.shape, dtype=torch.bool, device=c.device)

	for n in range(max_depth):
		dz = torch.where(alive, 2.0 * z * dz + 1.0, dz)
		z = torch.where(alive, z * z + c, z)
		escaped = alive & (z.abs() > PERIODIC_ESCAPE_R)
		z_esc = torch.where(escaped, z, z_esc)
		dz_esc = torch.where(escaped, dz, dz_esc)
		alive = alive & ~escaped
		if not bool(alive.any()):
			break

	zmag = z_esc.abs()
	dzmag = dz_esc.abs().clamp_min(1e-20)
	dist = zmag * torch.log(zmag.clamp_min(1.0001)) / dzmag
	phase = PERIODIC_BETA * torch.log(dist.clamp_min(1e-30))
	final_image = 0.5 + 0.5 * torch.sin(_TWO_PI * phase)
	final_image[alive] = 1.0              # never escaped -> in the set
	return final_image

def mandelbrotTensor(imag_values, real_values, max_depth, target='smooth', precision=64):
	# Combine real and imaginary parts into a complex tensor at the requested precision.
	# precision=64 (default) keeps the deep accuracy this project relies on; 32 is ~5x
	# faster on GPU and fine for shallow renders.
	real_dtype, complex_dtype = _dtypes(precision)
	c = (real_values + 1j * imag_values).to(complex_dtype)

	if target == 'periodic':
		return _mandelbrot_periodic_tensor(c, max_depth)

	z = torch.zeros_like(c)

	mask = torch.ones(c.shape, dtype=torch.bool, device=c.device)

	final_image = torch.zeros(c.shape, dtype=real_dtype, device=c.device)

	for n in range(max_depth):
		z = z**2 + c
		escaped = torch.abs(z) > 2
		mask = ~escaped & mask
		# print(n, smoothMandelbrot(n), torch.tensor([smoothMandelbrot(n)], dtype=torch.float64).cuda().cpu().numpy()[0])
		final_image[mask] = smoothMandelbrot(n)

	# iteration_count = torch.where(iteration_count==0, max_depth, iteration_count)
	final_image[torch.abs(z) <= 2] = 1.0 # all points that never escaped should be set to full white
	return final_image


class MandelbrotDataSet(Dataset):
	""" 
	Creates a dataset of randomized points and their calculated mandelbrot values.
  
	Parameters: 
	size (int): number of randomized points to generate
	max_depth (int): Maximum number of recursive steps before deciding\
	  whether the value is in the mandelbrot set
	xmin (float): minimum x value for points
	xmax (float): maximum x value for points
	ymin (float): minimum y value for points
	ymax (float): maximum y value for points
	target (str): which target to fit, 'smooth' (legacy) or 'periodic' (periodic\
	log-distance, ported from fractalsearch). 'periodic' wants a higher max_depth\
	(e.g. 200) to resolve exterior points. See the module-level notes.
	precision (int): GPU compute precision for the target, 64 (default, deep accuracy)\
	or 32 (~5x faster, fine for shallow renders/datasets).
	"""
	def __init__(self, size=1000, loadfile=None, max_depth=50, xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX,  dtype=torch.float32, gpu=False, target='smooth', precision=64):
		self.inputs = []
		self.outputs = []
		self.target = target
		if loadfile is not None:
			self.load(loadfile)
		else:
			print("Generating Dataset")
			if not gpu:
				for _ in tqdm(range(size)):
					x = random.uniform(xmin, xmax)
					y = random.uniform(ymin, ymax)
					self.inputs.append(torch.tensor([x, y]))
					self.outputs.append(torch.tensor(mandelbrot(x, y, max_depth, target=target)))
				self.inputs = torch.stack(self.inputs)
				self.outputs = torch.stack(self.outputs)
			else:
				X = (xmin - xmax) * torch.rand((size), dtype=dtype, device=device) + xmax
				Y = (ymin - ymax) * torch.rand((size), dtype=dtype, device=device) + ymax
				self.inputs = torch.stack([X, Y], dim=1).cpu()
				self.outputs = mandelbrotTensor(Y, X, max_depth, target=target, precision=precision).cpu()

		self.start_oversample(len(self.inputs))


	def __getitem__(self, i):
		if (i >= len(self.inputs)):
			ind = self.oversample_indices[i - len(self.inputs)]
			return self.inputs[ind], self.outputs[ind], ind.item()
		return self.inputs[i], self.outputs[i], i


	def __len__(self):
		return len(self.inputs) + len(self.oversample_indices)

	def start_oversample(self, max_size):
		self.max_size = max_size
		self.oversample_indices = torch.tensor([], dtype=torch.long)
		self.oversample_buffer = torch.tensor([], dtype=torch.long)
	
	def update_oversample(self):
		self.oversample_indices = self.oversample_buffer[:self.max_size]
		self.oversample_buffer = torch.tensor([], dtype=torch.long)

	def add_oversample(self, indices):
		indices = indices[indices < len(self.inputs)] # remove duplicates
		self.oversample_buffer = torch.cat([self.oversample_buffer, indices], 0)

	def save(self, filename):
		import os
		os.makedirs("./data", exist_ok=True)
		torch.save(self.inputs, './data/'+filename+'_inputs.pt')
		torch.save(self.outputs, './data/'+filename+'_outputs.pt')

	def load(self, filename):
		self.inputs = torch.load('./data/'+filename+'_inputs.pt')
		self.outputs = torch.load('./data/'+filename+'_outputs.pt')
