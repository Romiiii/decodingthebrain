import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


#3D data matrix (trial x channel x time)
DATA_PATH = ".\\DATA\\train_subject01.mat"
MODEL_STORE_PATH = ".\\models\\"

mat = sio.loadmat(DATA_PATH)

data = mat["X"]

mean = data.mean()
std = data.std()

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
print("um hello")
# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])

#
# #
# # # MNIST dataset
# # train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
# # test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
#
# # Try to reshape it to [batch, channel, width, height].
#
# class BrainDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, file_name, transform=None):
#         """
#         Args:
#             file_name (string): Path to the data file (X is input y is labels)
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.data = sio.loadmat(file_name)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data['y'])
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         # img_name = os.path.join(self.root_dir,
#         #                         self.landmarks_frame.iloc[idx, 0])
#         # image = io.imread(img_name)
#         #
#         # landmarks = self.landmarks_frame.iloc[idx, 1:]
#         # landmarks = np.array([landmarks])
#         # landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'trial': data['X'], 'label': data['y']}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

