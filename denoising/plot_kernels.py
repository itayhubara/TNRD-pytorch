import torch
import matplotlib.pyplot as plt

#w1=torch.load('/home/itayh/Tec/xUnit/denoising/weight_rand_init.pt')
#wr=torch.load("/home/itayh/Tec/xUnit/denoising/weight2_rand_init.pt")
w1=torch.load('/home/itayh/Tec/xUnit/denoising/weight_init_rot.pt')
wr=torch.load("/home/itayh/Tec/xUnit/denoising/weight2_init_rot.pt")
w1=w1.cpu().detach().numpy()
wr=wr.cpu().detach().numpy()

fig, axs = plt.subplots(4, 2,figsize=(5,20))
fig.tight_layout() 
axs[0, 0].imshow(w1[0,0])
axs[0, 0].set_title('k')
axs[0, 1].imshow(wr[0,0])
axs[0, 1].set_title('rot180(k)')
axs[1, 0].imshow(w1[1,0])
axs[1, 1].imshow(wr[1,0])
axs[2, 0].imshow(w1[2,0])
axs[2, 1].imshow(wr[2,0])
axs[3, 0].imshow(w1[3,0])
axs[3, 1].imshow(wr[3,0])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()
