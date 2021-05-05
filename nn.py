import scipy.io as sio
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from main import *
from os import listdir
from os.path import isfile, join

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

      #print(type(self.dataX))
      new_data = []
      # Reduce the dimensionality
      for i_data in range(len(self.dataX)):
          data = self.dataX[i_data]
          avg_sensor = averageSensorLocal(data, 3)
          max_pool_avg_sensor = maxPoolSensors(avg_sensor, 10)
          new_data.append(max_pool_avg_sensor)
      self.dataX = np.array(new_data)

      print("The dimensionality of the data is:", self.dataX.shape)
      # (100, 34, 38)

      if Normalize:
        mean = self.dataX.mean()
        std = self.dataX.std()
        self.dataX = (self.dataX - mean) / std

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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=3))
        self.drop_out = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2304, 4608)
        self.fc2 = nn.Linear(4608, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 2)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

# Try to reshape it to [batch, channel, width, height].
# (trial, channel, time)
# (594, 306, 375)

# (594, 102, 375)

DATA_PATH = ".\\DATA\\TRAIN\\"
MODEL_STORE_PATH = ".\\MODELS\\"

onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
#3D data matrix (trial x channel x time)
train = []
label = []
for f in onlyfiles:
    mat = sio.loadmat(DATA_PATH + f)
    train.extend(mat["X"])
    label.extend(mat["y"])

# Hyperparameters
num_epochs = 3
num_classes = 2
# Total Records (9414) 
# Divisible by: 1, 2, 3, 6, 9, 18, 523, 1046, 
# 1569, 3138, 4707, 9414
batch_size = 18
learning_rate = 0.001

# Generate dataset from data
train_dataset = BrainDataset(train, label)

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
        total = label.size(0)
        _, predicted = torch.max(outputs.data, 1) 
        correct = (predicted == label).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 5 == 0:
            print("Predicted", predicted)
            print("Correct", label)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

current_time = time.strftime("%d_%m_%y_%H_%M", time.localtime())
output_model_name = "decodebrain%s.model" % current_time
torch.save(model.state_dict(), MODEL_STORE_PATH + output_model_name )
