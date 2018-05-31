from train import trainMain
from PrepPC import PreparePointCloud
from plotter import plotModels
from plotter import plotModelsWithLabel

import numpy as np


def XYZcnnMain():
    NUM_POINTS = 128
    XYZ_point_cloud, labels, titles = PreparePointCloud(NUM_POINTS)
    
    #plotModels(XYZ_point_cloud, labels, titles, 10, NUM_POINTS)
    #plotModelsWithLabel(XYZ_point_cloud, labels, titles, 10, NUM_POINTS, 0)

    trainMain(XYZ_point_cloud, labels)
    eval()



XYZcnnMain();