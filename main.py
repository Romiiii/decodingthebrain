import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import numpy as np
#3D data matrix (trial x channel x time) 
mat = sio.loadmat(".\\DATA\\train_subject01.mat")

# The input is data of 1 trial
# Averaging the three sensors that are close to each other
# Since the 306 sensors are grouped, three at a time, in 102 locations
# step_size gets multiplied by 3, since sensors are grouped in trios
# (306 x 375)
#
def averageSensorLocal(trial_data, step_size):
    avg_sensor = []
    step_size *= 3
    for iSensor in range(0,len(trial_data),step_size):
    #for iTime in len(trial_data[0]):
        total = 0
        for i in range(step_size):
            total += trial_data[iSensor + i]
        avg = total / step_size
        avg_sensor.append(avg)
    return avg_sensor

# The input is data of 1 trial
# Preserve the peaks of the data
def maxPoolSensors(trial_data, step_size):
    max_pool_sensor = []
    for i_sensor in range(len(trial_data)):
        new_sensor_data = []
        sensor_data = trial_data[i_sensor]
        for i_time in range(0, len(sensor_data), step_size):
            max_value = max(sensor_data[i_time:i_time+step_size])
            new_sensor_data.append(max_value)
        max_pool_sensor.append(new_sensor_data)
    return max_pool_sensor


def plotData():
    #print(mat)
    axes = []
    number_of_plots = 5
    fig, axs = plt.subplots(number_of_plots)
    fig.suptitle('Visualizing Sensor Data')

    for j in range(number_of_plots):
        avg_sensor = averageSensorLocal(mat['X'][j], 3)
        max_pool_avg_sensor = maxPoolSensors(avg_sensor, 10)
        print(len(max_pool_avg_sensor), len(max_pool_avg_sensor[0]))

        for i in range(len(avg_sensor)):
            y = max_pool_avg_sensor[i]
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


