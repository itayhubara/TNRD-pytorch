import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from .modules.activations import *
import torchvision
from .modules.idct2_weight import gen_dct2
import numpy as np
import math

_NRBF=63 
_DCT=False
_TIE = False
_BETA = False 
_C1x1 = False
__all__ = ['g_tnrd','d_tnrd','TNRDlayer']

def initialize_weights(net, scale=1.):
    if not isinstance(net, list):
        net = [net]
    for layer in net:
        for m in layer.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, TNRDConv2d) or isinstance(m, TNRDlayer):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if not _TIE and isinstance(m, TNRDlayer):
                    nn.init.kaiming_normal_(m.weight2, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block                  
                if m.bias is not None and not isinstance(m, TNRDlayer):
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)
            

def init_model_param(model,num_reb_kernels=63,filter_size=5,stage=8,init_weight_dct=_DCT):
    w0 = np.load('w0_orig.npy')
    w0 = np.histogram(np.random.randn(1000)*0.02,num_reb_kernels-1)[1] if _NRBF != 63 else w0
    #
    means=np.linspace(-310,310,num_reb_kernels)
    precision=0.01
    NumW = num_reb_kernels
    step = 0.2
    delta = 10
    
    D = np.arange(-delta+means[0],means[-1]+delta,step)
    D_mu = D.reshape(1,-1).repeat(NumW,0) - means.reshape(-1,1).repeat(D.size,1)
    offsetD = D[1]
    nD = D.size
    G = np.exp(-0.5*precision*D_mu**2)
    filtN =  filter_size**2 - 1
    m = filter_size**2 - 1
    
    ww = np.array(w0).reshape(-1,1).repeat(filtN,1)
    cof_beta = np.eye(m,m)
    #x0 = zeros(length(cof_beta(:)) + 1 + filtN*mfs.NumW, stage);
    theta = [10, 5]+ np.ones(stage-2).tolist()
    pp = [math.log(1.0)]+ (math.log(0.1)*np.ones(stage-1)).tolist()
    # beta = [log(1) log(0.1)*ones(1,stage-1)];
    i=-1
    #import pdb; pdb.set_trace()
    for module in model.modules():
        if isinstance(module,TNRDlayer):
            i+=1
            init_layer_params(module,cof_beta, pp[i], ww*theta[i],init_weight_dct)

def init_layer_params(m,beta,p,wt,init_weight_dct):
    
    with torch.no_grad():
        if init_weight_dct:
            m.weight.copy_(torch.Tensor(beta))  
        else: 
            n = m.kernel_size**2 * m.in_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))     
            if not _TIE and isinstance(m, TNRDlayer):
                weight_rot180 = torch.rot90(torch.rot90(m.weight.data.detach(), 1, [2, 3]),1,[2,3])
                m.weight2.data.copy_(weight_rot180)   
                #m.weight2.data.normal_(0, math.sqrt(2. / n))             
            #initialize_weights(m,0.02)
        m.alpha.copy_(torch.Tensor([p]))
        if m.act.weight.shape[-1]==24:
            m.act.weight.copy_(torch.Tensor(wt).unsqueeze(1))    


class TNRDConv2d(nn.Conv2d):
    """docstring for TNRDConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=1, padding=2, dilation=1, groups=1, bias=True):
        super(TNRDConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.act = RBF(63,self.in_channels,triangular)
        self.alpha=nn.Parameter(torch.Tensor([0.9]))
        self.beta=nn.Parameter(torch.Tensor([1]))
        initialize_weights_dct([self], 0.02)
        self.pad_input=torchvision.transforms.Pad(5,padding_mode='edge')
        #initialize_weights([self.act], 0.00002)
        self.counter = 0 
    def forward(self,input):
        self.counter+=1
        u,f=input
        up = self.pad_input(u)
        ur = up.repeat(1,self.in_channels,1,1)
        #
        output1 = F.conv2d(ur, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output,_ = self.act(output1)
        weight_rot180 = torch.rot90(torch.rot90(self.weight, 1, [2, 3]),1,[2,3])
        #
        output = F.conv2d(output, weight_rot180, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = F.pad(output,(-5,-5,-5,-5))
        #import pdb; pdb.set_trace()
        if self.counter%100==0:
            print(self.alpha,self.beta.max(),self.beta.min(),self.act.weight.max(),self.act.weight.min(),output.sum(1,keepdim=True).max())
        output = u-self.beta*output.sum(1,keepdim=True)-self.alpha*(u-f)
        return output,f

class TNRDlayer(nn.Module):
    """docstring for TNRDConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(TNRDlayer, self).__init__()
        self.in_channels = in_channels
        self.stride=stride
        self.groups=groups
        self.bias=bias
        self.padding=padding
        self.stride=stride
        self.dilation=dilation
        self.kernel_size=kernel_size
        self.act = RBF(_NRBF,self.in_channels,gaussian)
        self.alpha=nn.Parameter(torch.Tensor([0.9]))
        self.beta=nn.Parameter(torch.zeros([1,self.in_channels,1,1]))
        if _DCT:
            self.weight = nn.Parameter(torch.zeros([in_channels,in_channels]))
        else:
            self.weight = nn.Parameter(torch.zeros([in_channels,1,kernel_size,kernel_size]))    
            if not _TIE:
                self.weight2 = nn.Parameter(torch.zeros([in_channels,1,kernel_size,kernel_size]))  
        if _C1x1:
            self.weight_1x1=nn.Parameter(torch.zeros([in_channels,in_channels,1,1]))  

        #initialize_weights_dct([self], 0.02)
        self.register_buffer('dct_filters',torch.tensor(gen_dct2(kernel_size)[1:,:]).float())
        
        self.pad_input=torchvision.transforms.Pad(12,padding_mode='symmetric')
        #initialize_weights([self.act], 0.00002)
        self.counter = 0 

    def forward(self,input):
        self.counter+=1
        u,f=input
        for it in range(1):
            up = self.pad_input(u)
            ur = up.repeat(1,self.in_channels,1,1)
            #import pdb; pdb.set_trace()
            if _DCT:
                K=self.weight.matmul(self.dct_filters)
                K = K.div(torch.norm(K,dim=1,keepdim=True)+2.2e-16).view(self.kernel_size**2-1,1,self.kernel_size,self.kernel_size)
            else:
                K = self.weight    
            
            output1 = F.conv2d(ur, K, None, self.stride, self.padding, self.dilation, self.groups)
            output,_ = self.act(output1)
            weight_rot180 = torch.rot90(torch.rot90(K, 1, [2, 3]),1,[2,3]) if _TIE else self.weight2
            if _C1x1:
                output = F.conv2d(output, self.weight_1x1, None, self.stride, self.padding, self.dilation)
            output = F.conv2d(output, weight_rot180, None, self.stride, self.padding, self.dilation, self.groups)
            output = F.pad(output,(-8,-8,-8,-8))
            #
            if self.counter%500==0:
                print(self.alpha,self.beta.max(),self.beta.min(),self.act.weight.max(),self.act.weight.min(),output.sum(1,keepdim=True).max())
            beta = self.beta if _BETA else 1
            u = u-output.mul(beta).sum(1,keepdim=True)-self.alpha.exp()*(u-f)
        return u,f

class GenBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=5, bias=True):
        super(GenBlock, self).__init__()
        self.conv = TNRDConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2), bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        initialize_weights([self.conv, self.bn], 0.02)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class DisBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True, normalization=False):
        super(DisBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

        initialize_weights([self.conv1, self.conv2], 0.1)

        if normalization:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        return x

class Generatorv1(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks):
        super(Generatorv1, self).__init__()
        filter_size=5
        # image to features
        in_channels=filter_size**2
        
        #self.crop_output=torchvision.transforms.CenterCrop(50)
        self.image_to_features = TNRDConv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=filter_size, groups=in_channels)
        # features
        blocks = []
        for _ in range(gen_blocks):
            blocks.append(TNRDConv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=filter_size, bias=False,groups=in_channels))
        self.features = nn.Sequential(*blocks)

        # features to image
        self.features_to_image = TNRDConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=filter_size,groups=in_channels)
        initialize_weights([self.features_to_image], 0.02)
        self.counter=0
    def forward(self, x):
        self.counter+=1
        x = self.image_to_features([x,x])
        x = self.features(x)
        x = self.features_to_image(x)
        return x[0]

class Generator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks):
        super(Generator, self).__init__()
        filter_size=5
        # image to features
        in_channels=(filter_size**2-1)
        
        #self.crop_output=torchvision.transforms.CenterCrop(50)
        self.image_to_features = TNRDlayer(in_channels=in_channels, out_channels=in_channels,groups=in_channels)
        # features
        blocks = []
        for _ in range(gen_blocks):
            blocks.append(TNRDlayer(in_channels=in_channels, out_channels=in_channels, bias=False,groups=in_channels))
        self.features = nn.Sequential(*blocks)
        
        #self.features = []
        #for _ in range(gen_blocks):
        #    self.features.append(TNRDlayer(in_channels=in_channels, out_channels=in_channels, bias=False,groups=in_channels))
        
        # features to image
        self.features_to_image = TNRDlayer(in_channels=in_channels, out_channels=in_channels, kernel_size=5,groups=in_channels)
        #initialize_weights([self.features_to_image], 0.02)
        self.counter=0
        init_model_param(self,num_reb_kernels=_NRBF,filter_size=5,stage=gen_blocks+2)

    def forward(self, x):
        
        self.counter+=1
        x = self.image_to_features([x,x])
        x = self.features(x)
        x = self.features_to_image(x)
        return x[0]

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks):
        super(Discriminator, self).__init__()

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=False)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=False))
        self.features = nn.Sequential(*blocks)

        # classifier
        self.classifier = nn.Conv2d(in_channels=num_features * min(pow(2, dis_blocks - 1), 8), out_channels=1, kernel_size=4, padding=0)

    def forward(self, x):
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(start_dim=1).mean(dim=-1)
        return x

class SNDiscriminator(nn.Module):
    def __init__(self, in_channels, num_features, gen_blocks, dis_blocks):
        super(SNDiscriminator, self).__init__()

        # image to features
        self.image_to_features = DisBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=True)

        # features
        blocks = []
        for i in range(0, dis_blocks - 1):
            blocks.append(DisBlock(in_channels=num_features * min(pow(2, i), 8), out_channels=num_features * min(pow(2, i + 1), 8), bias=False, normalization=True))
        self.features = nn.Sequential(*blocks)

        # classifier
        self.classifier = SpectralNorm(nn.Conv2d(in_channels=num_features * min(pow(2, dis_blocks - 1), 8), out_channels=1, kernel_size=4, padding=0))

    def forward(self, x):
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(start_dim=1).mean(dim=-1)
        return x

def g_tnrd(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', 3)
    config.setdefault('dis_blocks', 5)

    return Generator(**config)

def d_tnrd(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('gen_blocks', 8)
    config.setdefault('dis_blocks', 5)

    return Discriminator(**config)