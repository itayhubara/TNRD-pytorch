import logging.config
import os
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from os import path, makedirs
from skimage import color
import torch.nn.functional as F

def mkdir(save_path):
    if not path.exists(save_path):
        makedirs(save_path)

def make_image_grid(x, nrow, padding=0, pad_value=0):
    x = x.clone().cpu().data
    grid = make_grid(x, nrow=nrow, padding=padding, normalize=True, scale_each=False, pad_value=pad_value)
    return grid

def tensor_to_image(x):
    #ndarr = x.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr = x.squeeze().add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return image

def plot_image_grid(x, nrow, padding=0):
    grid = make_image_grid(x=x, nrow=nrow, padding=padding).permute(1, 2, 0).numpy()
    plt.imshow(grid)
    plt.show()

def save_image(x, path, size=None):
    scale=2
    #test_data_path='/home/itayh/Tec/Datasets/SR_TNRD/xunit_format/val/img/'
    #image_name = path.split('/')[-1]
    #input_d = Image.open(test_data_path+image_name).convert('RGB')
    #input_d = color.rgb2ycbcr(input_d)
    #x_im_d=x.squeeze().add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
    #input_d[:,:,0]=x_im_d
    #input_rgb=color.ycbcr2rgb(input)
    #input_rgb = torch.tensor(input_rgb).mul(255).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    #image = Image.fromarray(input_rgb.astype(np.uint8))
    #if size:
    #    image = image.resize((size, size), Image.NEAREST)
    #image.save(path)

    test_data_path='/home/itayh/Tec/Datasets/SR_TNRD/xunit_format/val/img_x%d/'%scale
    image_name = path.split('/')[-1]
    input = Image.open(test_data_path+image_name).convert('RGB')
    input = color.rgb2ycbcr(input)
    hh,ww,_=input.shape
    input = torch.tensor(input).permute(2, 0, 1).view(1,3,hh,ww)
    input = F.interpolate(input, scale_factor=scale, mode='bicubic', align_corners=False)
    input=input.squeeze().permute(1, 2, 0).add_(0.5).clamp_(0, 255)
    
    x_im=x.squeeze().add(0.5).clamp(0, 255).floor()
    input[:,:,0]=x_im
    input_rgb=color.ycbcr2rgb(input.to('cpu').numpy())
    input_rgb = torch.tensor(input_rgb).mul(255).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(input_rgb.astype(np.uint8))
    if size:
        image = image.resize((size, size), Image.NEAREST)
    image.save(path)

def save_image_grid(x, path, nrow=8, size=None):
    grid = make_image_grid(x, nrow)
    save_image(grid, path, size=size)

def setup_logging(log_file='log.txt', resume=False, dummy=False):
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=log_file,
                            filemode=file_mode)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def average(lst):
    return sum(lst) / len(lst)

def rgb2yc(img):
    rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    rlt = rlt.round()
    return rlt

def compute_psnr(x, y, scale):
    #x = rgb2yc(tensor_to_image(x)).astype(np.float64)
    #y = rgb2yc(tensor_to_image(y)).astype(np.float64)
    
    #x = color.rgb2ycbcr(tensor_to_image(x)).astype(np.float64)[0]
    #y = color.rgb2ycbcr(tensor_to_image(y)).astype(np.float64)[0]
    x=x.squeeze().clamp_(0, 255).round().float()
    y=y.squeeze().clamp_(0, 255).round().float()
    x = x[round(scale):-round(scale), round(scale):-round(scale)]
    y = y[round(scale):-round(scale), round(scale):-round(scale)]
    mse = ((x - y) ** 2).mean()
    print(20 * math.log10(255.0 / math.sqrt(mse)))
    return 20 * math.log10(255.0 / math.sqrt(mse))

if __name__ == "__main__":
    print('None')

