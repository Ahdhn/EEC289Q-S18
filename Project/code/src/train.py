import tensorflow as tf
import numpy as np
import math

from util import KNN

def Var(name, shape, init, use_fp16=False, train = True):
    #create var stored on cpu mem
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer= init, dtype= dtype, trainable= train)
    return var 

def variable_weight_decay(name, shape,stddev, weight_decay, use_xavier=True):
    if use_xavier:
        init = tf.contrib.layers.xavier_initializer()
    else:
        init = tf.truncated_normal_initializer(stddev = stddev)
    var = Var(name, shape, init)
    if weight_decay is not None:
        decay = tf.multiply(tf.nn.l2_loss())



def conv2d(inputs, 
           num_output_channels,
           kernel_size,
           scope,
           stride = [1,1],
           padding = 'SAME',
           use_xavier = True,
           stddev = 1e-3,
           weight_decay = 0.0,
           activation_fn = tf.nn.relu,
           bn = False,
           decay = None,
           traning= None,
          dist = False):
    #input is 4-D tensor 
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
        kernel = _va

def createModel(points_pl, decay, adj_matrix):
    #input_points is a placeholder 
    batch_size = points_pl.get_shape()[0].value
    num_pts = points_pl.get_shape()[1].value    
    k=20

    edgeFeatures = edge_features(points_pl, adj_matrix, k)

    #with tf.variable_scope('transform_net1') as sc:
    
    net =  tf.nn.conv2d(edge_features,64)


    

def testModel():
    #testing the model by generating random points
    batch_size=2
    num_pts = 124
    dim=3

    #generate random points, assigne labels to them as 1 and 0
    input_points = np.random.rand(batch_size, num_pts, dim)
    labels = np.random(rand, batch_size)
    labels[labels >=0.5]=1
    labels[labels <0.5]=0
    #cast to int 
    labels = labels.astype(np.int32) 

    with tf.Graph().as_default():#not needed but it is a good practice 
        #placeholder for the points 
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, num_pts, dim))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        knn_graph =  KNN(pointcloud_pl=points_pl, k=20)





def Model(input_points, 
          labels,
          batch_size,
          dim,
          num_pts):

    with tf.Graph().as_default():#not needed, but it is a good practice 
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, num_pts, dim))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        knn_graph =  KNN(pointcloud_pl=points_pl, k=20)



def trainMain(XYZ_point_cloud, labels): 

    
    print("Training");

    batch_size = 2
    num_point = 1024
    pos_dim = 3
    max_epoch = 250
    learning_rate = 0.001
    momentum = 0.9
    optimizer = 'adam'
    decay_step = 200000
    decay_rate = 0.7
    max_num_point = 2048
    num_classes = 40


    Init_decay = 0.5
    decay_decay_rate = 0.5
    decay_decay_step = float(decay_step)
    decay_clip = 0.99

    Model(XYZ_point_cloud, labels, batch_size, pos_dim, num_point)

    

    tf.nn.conv2d()

if __name__ == "__main__":    

    if False:
        testModel()

    trainMain()