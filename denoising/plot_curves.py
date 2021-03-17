import torch
import matplotlib.pyplot as plt
import numpy as np

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#w1=torch.load('/home/itayh/Tec/xUnit/denoising/weight_rand_init.pt')
#wr=torch.load("/home/itayh/Tec/xUnit/denoising/weight2_rand_init.pt")
#losses=torch.load('/home/itayh/Tec/xUnit/denoising/results/2021-03-14_21-20-09/losses_e100.pt')

losses_tnrd=torch.load('/home/itayh/Tec/xUnit/denoising/results/2021-03-14_23-20-00/losses_e600.pt')
losses=torch.load('/home/itayh/Tec/xUnit/denoising/results/2021-03-15_08-43-20/losses_e600.pt')

import pdb; pdb.set_trace()
#losses['psnr'] = [25.552,25.612,25.672,25.682,25.692,25.72,25.74]
#losses['psnr'] += [25.74,25.75,25.74,25.76,25.77,25.79,25.80]
#plt.scatter([200*10*i for i in range(len(losses['psnr']))],losses['psnr'])
#plt.plot(smooth(np.array(losses['tnrd_loss']),50)[50:-50],'r')
#plt.plot(smooth(np.array(losses['G_recon'])/10,50)[50:-50],'g')
#plt.ylim(-1000,100)

#plt.plot(20*np.log(smooth(np.array(losses_tnrd['tnrd_loss']),50))[70:-50],'r')
#plt.plot(20*np.log(smooth(np.array(losses['tnrd_loss']),50))[70:-50],'g')
#plt.legend(['Regularized with TNRD loss','Without TNRD loss'])
#plt.show()


#plt.scatter([200*10*i for i in range(len(losses_tnrd['psnr']))],losses_tnrd['psnr'],color='red')
#plt.scatter([200*10*i for i in range(len(losses['psnr']))],losses['psnr'],color='green')
#plt.legend(['Regularized with TNRD loss','Without TNRD loss'])
#plt.show()
#
#
plt.plot(smooth(np.array(losses_tnrd['G_recon']),50)[50:-50],'r')
plt.plot(smooth(np.array(losses['G_recon']),50)[50:-50],'g')
plt.legend(['Regularized with TNRD loss','Without TNRD loss'])
plt.show()