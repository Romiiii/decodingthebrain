import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class BrainDataset(Dataset):
  def __init__(self, train, label, Normalize=True):
      """
      Args:
          file_name (string): Path to the data file (X is input y is labels)
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      self.dataX = train
      self.labels= label

      if Normalize:
        mean = train.mean()
        std = train.std()
        self.dataX = (train - mean) / std

  def __len__(self):
      return len(self.labels)

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      # img_name = os.path.join(self.root_dir,
      #                         self.landmarks_frame.iloc[idx, 0])
      # image = io.imread(img_name)
      #
      # landmarks = self.landmarks_frame.iloc[idx, 1:]
      # landmarks = np.array([landmarks])
      # landmarks = landmarks.astype('float').reshape(-1, 2)

      return self.dataX[idx], self.labels[idx][0]

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        #self.layer2 = nn.Sequential(
        #    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=3))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(301 * 370 * 32, 1000)
        self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
# Try to reshape it to [batch, channel, width, height].
# (594, 306, 375)

#3D data matrix (trial x channel x time)
DATA_PATH = ".\\DATA\\train_subject01.mat"
MODEL_STORE_PATH = ".\\models\\"

mat = sio.loadmat(DATA_PATH)
print(type(mat))

train = mat["X"]
print(train.shape)
label = mat["y"]

mean = train.mean()
std = train.std()

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 99
learning_rate = 0.001

#
# #
# MNIST dataset
train_dataset = BrainDataset(train,label)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (trial, label) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(trial.unsqueeze(1))
        loss = criterion(outputs, label)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

