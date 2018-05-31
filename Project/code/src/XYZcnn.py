from train import trainMain
from PrepPC import PreparePointCloud
from plotter import plotModels

import numpy as np


def XYZcnnMain():
    
    XYZ_point_cloud, labels, titles = PreparePointCloud()
    plotModels(XYZ_point_cloud, labels, titles, 10, 1024)

    trainMain(XYZ_point_cloud, labels)
    eval()



XYZcnnMain();