import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
#import tf_util

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/test_files.txt'))

def PreparePointCloud(NUM_POINT):
    print("PreparePointCloud")
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    test_file_idxs = np.arange(0, len(TEST_FILES))
    #np.random.shuffle(train_file_idxs)
   
    labelsName = provider.getDataFiles( \
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/shape_names.txt'))

    XYZ_point_cloud, labels, normals = provider.loadDataFile('../' + TRAIN_FILES[train_file_idxs[0]])
    for i in range(len(TRAIN_FILES) - 1):
        xyz, lab, norm = provider.loadDataFile('../' + TRAIN_FILES[train_file_idxs[i + 1]])
        XYZ_point_cloud = np.concatenate((XYZ_point_cloud, xyz))
        labels = np.concatenate((labels, lab))
        normals = np.concatenate((normals, norm))

    for i in range(len(TEST_FILES)):
        xyz, lab, norm = provider.loadDataFile('../' + TEST_FILES[test_file_idxs[i]])
        XYZ_point_cloud = np.concatenate((XYZ_point_cloud, xyz))
        labels = np.concatenate((labels, lab))    
        normals = np.concatenate((normals, norm))
    
    XYZ_point_cloud = XYZ_point_cloud[:,0:NUM_POINT,:]
    normals = normals[:,0:NUM_POINT,:]

    return XYZ_point_cloud, labels, labelsName, normals