import argparse
import glob
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch

def get_arguments():
    parser = argparse.ArgumentParser(description='super-resolution-image-prepare')
    parser.add_argument('--scale', default=4, type=int, help='scale')
    parser.add_argument('--dir', default='.', help='images directory')
    parser.add_argument('--output', default='.', help='output director')
    args = parser.parse_args()
    return args



def extract_image_paths(dir):
    paths=[]
    files=[]
    for file in os.listdir(dir):
        if file.endswith(".png") or file.endswith(".bmp"):
            paths.append(os.path.join(dir, file))
            files.append(file)
    return paths, files       
def main(args):
    args.output+='img_x%d/'%args.scale
    os.makedirs(args.output, exist_ok=True)
    image_paths,files_names = extract_image_paths(args.dir)
    for path,file in zip(image_paths,files_names):
        input = Image.open(path)
        x=torch.tensor(np.array(input)).unsqueeze(0).permute(0,3,1,2).float()
        new_h=(x.shape[2]//4)*4
        new_w=(x.shape[3]//4)*4
        x = x[:,:,:new_h,:new_w]
        im_orig = Image.fromarray(x.permute(0,2,3,1).squeeze().round().clamp(0,255).numpy().astype(np.uint8))
        im_orig.save('%s%s'%(args.output.replace('_x%d'%args.scale,''),file))
        r = F.interpolate(x, scale_factor=1/args.scale, mode='bicubic', align_corners=False)    
        im = Image.fromarray(r.permute(0,2,3,1).squeeze().round().clamp(0,255).numpy().astype(np.uint8))
        im.save('%s%s'%(args.output,file))
    print('Done')


if __name__=='__main__':
    args = get_arguments()
    main(args)