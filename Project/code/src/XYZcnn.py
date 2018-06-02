import tensorflow as tf 
import numpy as np

from train import trainMain
from PrepPC import PreparePointCloud
import pandas as pd

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from plotter import plotModels
#from plotter import plotModelsWithLabel

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def XYZcnnMain():
    NUM_POINTS = 2048
    XYZ_point_cloud, labels, titles, point_normals = PreparePointCloud(NUM_POINTS)
   

    #plotModels(XYZ_point_cloud, labels, titles, 1, NUM_POINTS)
    #plotModelsWithLabel(XYZ_point_cloud, labels, titles, 10, NUM_POINTS, 0)

    trainMain(XYZ_point_cloud=XYZ_point_cloud, labels=labels, 
              XYZ_point_notmals = point_normals)
    



XYZcnnMain();