from scipy.fftpack import dct, idct
import numpy as np 
import math

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

def gen_dct2(n):
    
    C = np.zeros([n**2,n**2])
    for i in range(n):
        for j in range(n):
          A=np.zeros([n,n])
          A[i,j]=1
          B = idct2(A)
          C[:,(j-1)*n + i] = B.reshape(-1)
    return C

