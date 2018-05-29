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

def PreparePointCloud(fn):
    print("PreparePointCloud")

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    XYZ_point_cloud, labels = provider.loadDataFile('../' + TRAIN_FILES[train_file_idxs[fn]])

    return XYZ_point_cloud, labels