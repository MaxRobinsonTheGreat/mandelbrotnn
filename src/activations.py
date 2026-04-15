import torch
import torch.nn as nn

class MexicanHatActivation(nn.Module):
    """
    Mexican Hat (Ricker) Wavelet Activation Function.
    
    Computes: 
      f(x) = (1 - x^2) * exp(-0.5 * x^2)
    """
    def __init__(self):
        super(MexicanHatActivation, self).__init__()
    def forward(self, x):
        return (1 - x**2) * torch.exp(-0.5 * x**2)

class MorletActivation(nn.Module):
    """
    Morlet Wavelet Activation Function.
    
    Computes: 
      f(x) = cos(x) * exp(-0.5 * x^2)
    """
    def __init__(self):
        super(MorletActivation, self).__init__()
        
    def forward(self, x):
        return torch.cos(x) * torch.exp(-0.5 * x**2)

class HaarActivation(nn.Module):
    """
    Haar Wavelet Activation Function.
    
    Implements the Haar wavelet:
      ψ(x) =  1,  if 0 ≤ x < 0.5;
             -1,  if 0.5 ≤ x < 1;
              0,  otherwise.
    
    Note: This activation is piecewise constant and not differentiable at the jump points.
    """
    def __init__(self):
        super(HaarActivation, self).__init__()
        
    def forward(self, x):
        # Initialize output with zeros.
        out = torch.zeros_like(x)
        # Define piecewise conditions.
        condition1 = (x >= 0) & (x < 0.5)
        condition2 = (x >= 0.5) & (x < 1)
        out = torch.where(condition1, torch.ones_like(x), out)
        out = torch.where(condition2, -torch.ones_like(x), out)
        return out
