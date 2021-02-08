import scipy.io as sio
import torch
mat = sio.loadmat(".\\DATA\\train_subject01.mat")
#print(mat)
#print(mat[0])
print(mat['X'][2][0][0])
torchInput = torch.randn(20, 16, 50, 100)
torchInput 
