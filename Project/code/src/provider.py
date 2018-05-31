import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:]
  label = f['label'][:]
  normal = f['normal'][:]
  return (data, label, normal)

def loadDataFile(filename):
  return load_h5(filename)