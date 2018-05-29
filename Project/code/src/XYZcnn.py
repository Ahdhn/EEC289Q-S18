from train import trainMain
from PrepPC import PreparePointCloud

def XYZcnnMain():    
    XYZ_point_cloud, labels = PreparePointCloud() 
    trainMain(XYZ_point_cloud, labels)
    eval()



XYZcnnMain();