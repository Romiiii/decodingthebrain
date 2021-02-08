import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import numpy as np
#3D data matrix (trial x channel x time) 
mat = sio.loadmat(".\\DATA\\train_subject01.mat")
def averageSensorLocal(trial_data):
    avg_sensor = []
    for iSensor in range(0,len(trial_data),3):
    #for iTime in len(trial_data[0]):
        avg = (trial_data[iSensor] + trial_data[iSensor+1] +
        trial_data[iSensor+2])/3
        avg_sensor.append(avg)
    return avg_sensor

#print(mat)
axes = []
fig, axs = plt.subplots(2)
fig.suptitle('Visualizing Sensor Data')

for j in range(2):
    avg_sensor = averageSensorLocal(mat['X'][j])
    for i in range(len(avg_sensor)):
        y = avg_sensor[i]    
        x = range(len(y))
        axs[j].set_title('Trial n:' + str(j) + " | Label is: " + str(mat['y'][j]))
        axs[j].plot(x,y)

#for j in range(5):
#    for i in range(len(mat['X'][j])):
#        y = mat['X'][j][i]
#        print(y.shape)
#        x = range(len(mat['X'][j][i]))
#        axs[j].set_title('Trial n:' + str(j) + " | Label is: " + str(mat['y'][j]))
#        axs[j].plot(x,y)

fig.tight_layout()    
plt.show()
torchInput = torch.randn(20, 16, 50, 100)
