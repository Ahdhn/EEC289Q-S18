import tensorflow as tf
import numpy as np
import math

from util import KNN

###### TF helper functions 
def VarCPU(name, shape, init, use_fp16=False, train = True):
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
    var = VarCPU(name, shape, init)
    if weight_decay is not None:
        decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', decay)
    return var
def batch_norm_dist(inputs, training, scope, moments_dims, decay):

    #batch normalization 
    #https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

    with tf.variable_scope(scope)as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = VarCPU('beta',[num_channels], init=tf.zeros_initializer())
        gamma = VarCPU('gamma',[num_channels], init=tf.ones_initializer())
        pop_mean = VarCPU('pop_mean', [num_channels], init=tf.zeros_initializer(),train=False)
        pop_var = VarCPU('pop_var', [num_channels], init=tf.ones_initializer(),train=False)

        def train_bn_op():
            batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name = 'moments')
            mdecay = decay if decay is not None else 0.9
            train_mean = tf.assign(pop_mean,pop_mean*mdecay +batch_mean*(1-mdecay))
            train_var = tf.assign(pop_var,pop_var*mdecay +batch_var*(1-mdecay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, gamma,1e-3)
        def test_bn_op():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var,beta, gamma, 1e-3)

        normed = tf.cond(training,train_bn_op,test_bn_op())
        return normed
def batch_norm(inputs, training, scope, moments_dim, decay):
    #batch normalization on conv maps 
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name = 'beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                           name = 'gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dim, name='moments')
        mdecay = decay if decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay = mdecay)

        #op that maintain moving averages of varaibles 
        ema_apply_op = tf.cond(training, 
                               lambda:ema.apply([batch_mean, batch_var]),
                               lambda:tf.no_op())
        #update moving average and retuen current batch's avg and var
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)

        return normed

###### TF layers creator function
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
           training= None,
           dist = False):
    #input is 4-D tensor 
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
        kernel = variable_weight_decay('weights', 
                                       shape =kernel_shape, 
                                       stddev=stddev, 
                                       weight_decay=weight_decay, 
                                       use_xavier=use_xavier)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1,stride_h, stride_w,1],
                               padding=padding)
        biases = VarCPU('biases',
                        [num_output_channels], 
                        tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)
        if bn:
            if dist:                
                outputs = batch_norm_dist(inputs=outputs,training=training,scope='bn',moments_dim=[0,1,2],decay=decay)
            else:
                outputs  = batch_norm(inputs=outputs,training=training,scope='bn',moments_dim=[0,1,2],decay=decay)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

def fully_connected(inputs, 
                    num_outputs, 
                    scope,
                    use_xavier = True,
                    stddev = 1e-3,
                    weight_decay= 0.0,
                    activation_fn = tf.nn.relu,
                    bn = False, 
                    decay = None, 
                    training = None, 
                    dist = False):
    #input is 2D tensor 
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = variable_weight_decay('weights', 
                                        shape= [num_input_units, num_outputs],
                                        stddev=stddev,
                                        weight_decay=weight_decay)
        outputs = tf.matmul(inputs,weights)
        biases = VarCPU('biases', [num_outputs],
                        tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs,biases)
        if bn:
            if dist:
                outputs = batch_norm_dist(inputs=outputs,training=training,scope='bn',moments_dim=[0,],decay=decay)                
            else:
                outputs = batch_norm(inputs=outputs,training=training,scope='bn',moments_dim=[0,],decay=decay)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

def max_pooling2d(inputs, 
                  kernel_size, 
                  scope, 
                  stride = [2,2],
                  padding = 'VALID'):
    #2d max pooling
    #input is 4D tensor 
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs, 
                                 ksize=[1,kernel_h,kernel_w,1],
                                 strides=[1,stride_h,stride_w,1],
                                 padding=padding,
                                 name=sc.name)
        return outputs
def dropout(inputs,
            training, 
            scope,
            keep_prod = 0.5,
            noise_shape = None):
    #dropout layer
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(training,
                          lambda:tf.nn.dropout(inputs,keep_prod,noise_shape),
                          lambda:inputs)
        return outputs

###### Operator 
def edgeConv(points_pl, knn_graph, k=20):
    #get the edge feature
    og_batch_size = points_pl.get_shape().as_list()[0]
    points_pl = tf.squeeze(points_pl)
    if og_batch_size == 1:
        points_pl = tf.expand_dims(points_pl,0)

    points_pl_central = points_pl 
    points_pl_shape = points_pl.get_shape()
    batch_size = points_pl_shape[0].value
    num_points = points_pl_shape[1].value
    num_dim = points_pl_shape[2].value

    idx = tf.range(batch_size)*num_points
    idx = tf.reshape(idx,[batch_size,1,1])

    points_pl_flat = tf.reshape(points_pl,[-1,num_dim])
    points_pl_neighbors = tf.gather(points_pl_flat, knn_graph+idx)
    points_pl_central = tf.expand_dims(points_pl_central, axis=-2)

    points_pl_central = tf.tile(points_pl_central,[1,1,k,1])

    edge_conv = tf.concat([points_pl_central, points_pl_neighbors-points_pl_central],axis=-1)
    return edge_conv

###### Model creation 
def Model(input_points, 
          labels,
          batch_size,
          dim,
          num_pts):

    with tf.Graph().as_default():#not needed, but it is a good practice 
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, num_pts, dim))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        knn_graph =  KNN(pointcloud_pl=points_pl, k=20)

def createModel(points_pl, training, knn_graph, decay=None):
    #input_points is a placeholder 
    batch_size = points_pl.get_shape()[0].value
    num_pts = points_pl.get_shape()[1].value    
    end_points = {}
    k=20

    #get features 
    edgeFeatures = edgeConv(points_pl=points_pl, knn_graph=knn_graph, k=k)

    
    #conv1 
    net = conv2d(inputs= edgeFeatures,
                 num_output_channels=64,
                 kernel_size=[1,1],
                 scope='xyzcnn1',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)
    net =tf.reduce_max(net, axis=-2, keepdims=True)
    net1 = net
    #print(net1)

    #conv2
    net = conv2d(inputs= edgeFeatures,
                 num_output_channels=64,
                 kernel_size=[1,1],
                 scope='xyzcnn2',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)
    net = tf.reduce_mean(net, axis=-2, keepdims=True)
    net2 = net
    #print(net2)

    #conv3
    net = conv2d(inputs= edgeFeatures,
                 num_output_channels=64,
                 kernel_size=[1,1],
                 scope='xyzcnn3',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)   
    net = tf.reduce_mean(net, axis=-2, keepdims=True)
    net3 = net
    #print(net3)
    #conv4
    net = conv2d(inputs= edgeFeatures,
                 num_output_channels=128,
                 kernel_size=[1,1],
                 scope='xyzcnn4',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)   
    net = tf.reduce_mean(net, axis=-2, keepdims=True)
    net4 = net
    #print(net4)

    #agg
    net = conv2d(inputs= tf.concat([net1, net2, net3, net4], axis=-1),
                 num_output_channels=1024,
                 kernel_size=[1,1],
                 scope='agg',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)   
    net = tf.reduce_mean(net, axis=-2, keepdims=True)
    agg = net
    #print(agg)

    ### MLP
    #reshape
    net = tf.reshape(net, [batch_size,-1])
    #print(net)

    #FC1    
    net = fully_connected(inputs = net,
                          num_outputs= 512,
                          scope = 'fc1',                          
                          bn=True,
                          decay= decay,
                          training = training)
    #print(net)

    #dp1
    net = dropout(inputs=net, 
                  training=training,
                  scope='dp1',
                  keep_prod=0.5)
    #print(net)

    #FC2
    net = fully_connected(inputs = net,
                          num_outputs= 256,
                          scope = 'fc2',                          
                          bn=True,
                          decay= decay,
                          training = training)
    #print(net)

    #dp2
    net = dropout(inputs=net, 
                  training=training,
                  scope='dp2',
                  keep_prod=0.5)
    #print(net)

    #FC3
    net = fully_connected(inputs = net,
                          num_outputs= 40,
                          scope = 'fc3',                          
                          bn=True,
                          decay= decay,
                          training = training)
    #print(net)

    return net, end_points
    
###### Testing 
def get_loss(pred, label, end_points):
    #pred: b*num_classes
    #labels: b
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss

def testModel():
    #testing the model by generating random points
    batch_size=2
    num_pts = 124
    dim=3

    #generate random points, assigne labels to them as 1 and 0
    input_points = np.random.rand(batch_size, num_pts, dim)
    labels = np.random.rand(batch_size)
    labels[labels >=0.5]=1
    labels[labels <0.5]=0
    #cast to int 
    labels = labels.astype(np.int32) 

    with tf.Graph().as_default():#not needed but it is a good practice 
        #placeholder for the points 
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, num_pts, dim))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        knn_graph =  KNN(pointcloud_pl=points_pl, k=20)
        pos, features = createModel(points_pl=points_pl,training=tf.constant(True),knn_graph=knn_graph)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {points_pl:input_points, labels_pl:labels}
            res1, res2 =sess.run([pos, features], feed_dict=feed_dict)
            print(res1.shape)
            print(res1)        
        
###### Training  
def trainMain(XYZ_point_cloud, labels):
    
    print("Training")

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

    if True:
        testModel()

    trainMain()