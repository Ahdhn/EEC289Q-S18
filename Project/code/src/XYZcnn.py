from train import trainMain
from PrepPC import PreparePointCloud

def XYZcnnMain():
    
    fn = 0    
    XYZ_point_cloud, labels = PreparePointCloud(fn) 
    trainMain(XYZ_point_cloud, labels)
    eval()



XYZcnnMain();