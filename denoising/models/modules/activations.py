import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F


# RBF Layer

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centers: the learnable centers of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, num_func, num_filters, basis_func):
        super(RBF, self).__init__()
        self.num_func = num_func
        self.register_buffer('centers', torch.tensor(np.linspace(-310,310,num_func)).float())
        #self.centers= nn.Parameter( torch.tensor(np.linspace(-310,310,num_func)).float())
        self.num_filters = num_filters
        self.weight = nn.Parameter(torch.Tensor(num_func, 1, num_filters))
        self.gamma=10 
        #self.gamma = nn.Parameter(torch.tensor([10.0]))
        self.basis_func = basis_func  
        self.int_basis_func = erf_func
        #self.bn=nn.BatchNorm2d(self.num_filters)
        #self.relu=nn.ReLU()
        #self.reset_parameters()

    #def reset_parameters(self):
    #    with torch.no_grad():
    #        nn.init.normal_(self.weight, 0, 1)
    #        self.weight.div_(self.weight.var())
        #import pdb; pdb.set_trace()
    #    nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        #input=self.bn(input)
        #input=F.layer_norm(input,input.size()[1:])
        #return self.relu(input),self.relu(input)
        size = [self.num_func]+list(input.shape)
        x = input.expand(size)
        c = self.centers.view(-1,1,1,1,1)
        weight = self.weight.view(-1,1,self.num_filters,1,1)
        if self.basis_func==gaussian:
            distances = (x - c).div(self.gamma)
            return self.basis_func(distances).mul(weight).sum(0),self.int_basis_func(distances,self.gamma).mul(weight).sum(0)
        else:    
            distances = (x - c).abs()
            return self.basis_func(distances,self.gamma).mul(weight).sum(0),0



# RBFs

def gaussian(alpha):
    phi = torch.exp(-0.5*alpha.pow(2))
    return phi

def erf_func(alpha,gamma):
    phi = gamma*math.sqrt(math.pi/2)*torch.erf(alpha.div(math.sqrt(2)))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi
def triangular(alpha,gamma):
    out = 1-alpha.div(gamma)
    out[alpha>gamma]=0
    return out
def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
