import numpy as np
import os
'''
This function contains post-processed datasets
'''
class datasets(object):
    def __init__(*arg):
        datasets.description = "To get datasets for the model"
    def dataV1(*arg):
        data_path = os.getcwd()+"\\dataset\\Training_set\\dataset_train.npy"
        data = np.load(data_path,allow_pickle=True).tolist()
        return data['xdata'],data['ydata']
    
    def dataV2(*arg):
        data_path = os.getcwd()+"\\dataset\\Training_set\\dataset_trainV01.npy"
        data = np.load(data_path,allow_pickle=True).tolist()
        return data['xdata'],data['ydata']
    def dataV3(*arg):
        print('Data version is not found ....')
# data = datasets
# xdata,ydata = data.dataV1()
# print('___')