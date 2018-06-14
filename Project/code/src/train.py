import tensorflow as tf
import numpy as np
import math
import shutil
import os
import sys
import random
from util import KNN
from util import dotProduct
from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#filter out warning 
LOG_DIR = 'log_test'
if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)
else:
    shutil.rmtree(LOG_DIR)
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')


############################################################################
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
    var = VarCPU(name = name,shape= shape,init= init)
    if weight_decay is not None:
        decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', decay)
    return var
def batch_norm_dist(inputs, training, scope, moments_dims, decay):

    #batch normalization 
    #https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

    with tf.variable_scope(scope)as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = VarCPU(name = 'beta',shape = [num_channels], init=tf.zeros_initializer())
        gamma = VarCPU(name = 'gamma',shape = [num_channels], init=tf.ones_initializer())
        pop_mean = VarCPU(name = 'pop_mean', shape = [num_channels], init=tf.zeros_initializer(),train=False)
        pop_var = VarCPU(name = 'pop_var', shape= [num_channels], init=tf.ones_initializer(),train=False)

        def train_bn_op():
            batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name = 'moments')
            mdecay = decay if decay is not None else 0.9
            train_mean = tf.assign(pop_mean,pop_mean*mdecay +batch_mean*(1-mdecay))
            train_var = tf.assign(pop_var,pop_var*mdecay +batch_var*(1-mdecay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, gamma,1e-3)
        def test_bn_op():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var,beta, gamma, 1e-3)

        normed = tf.cond(training,train_bn_op,test_bn_op)
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

############################################################################
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
                                        use_xavier=use_xavier,
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

############################################################################
###### Operator 
def normConv(points_normals_pl, knn_graph, k=20, concat_points = True):
    #get the edge feature
    og_batch_size = points_normals_pl.get_shape().as_list()[0]
    points_normals_pl = tf.squeeze(points_normals_pl)
    if og_batch_size == 1:
        points_normals_pl = tf.expand_dims(points_normals_pl,0)

    points_normals_pl_central = points_normals_pl 
    points_normals_pl_shape = points_normals_pl.get_shape()
    batch_size = points_normals_pl_shape[0].value
    num_points = points_normals_pl_shape[1].value
    num_dim = points_normals_pl_shape[2].value

    idx = tf.range(batch_size)*num_points
    idx = tf.reshape(idx,[batch_size,1,1])

    points_normals_pl_flat = tf.reshape(points_normals_pl,[-1,num_dim])
    points_normals_pl_neighbors = tf.gather(points_normals_pl_flat, knn_graph+idx)
    points_normals_pl_central = tf.expand_dims(points_normals_pl_central, axis=-2)

    points_normals_pl_central = tf.tile(points_normals_pl_central,[1,1,k,1])

    norm_conv = tf.reduce_sum(tf.multiply(points_normals_pl_central,points_normals_pl_neighbors), 3,keep_dims=True)
    
    if concat_points:
        norm_conv = tf.concat([points_normals_pl_central, norm_conv],axis=-1)
    
    return norm_conv


def edgeConv(points_pl, knn_graph, k=20, concat_points=True):
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
    

    if concat_points:
        edge_conv = tf.concat([points_pl_central, points_pl_neighbors-points_pl_central],axis=-1)
    else:
        edge_conv = points_pl_neighbors-points_pl_central   

    return edge_conv

############################################################################
###### Model creation 
def createModel_Dyn(points_pl,                    
	                training,
	                knn10_graph,
	                knn20_graph,
	                knn30_graph,
	                knn40_graph,                    
                    points_normals_pl = None,
	                knn50_graph=None,
	                knn60_graph=None,
	                decay=None):

    #input_points is a placeholder 
    batch_size = points_pl.get_shape()[0].value
    num_pts = points_pl.get_shape()[1].value       
    

    #get features     
    edgeFeatures10 = edgeConv(points_pl=points_pl, knn_graph=knn10_graph, k=10, concat_points = False)  
    edgeFeatures20 = edgeConv(points_pl=points_pl, knn_graph=knn20_graph, k=20, concat_points = False)
    edgeFeatures30 = edgeConv(points_pl=points_pl, knn_graph=knn30_graph, k=30, concat_points = False)
    edgeFeatures40 = edgeConv(points_pl=points_pl, knn_graph=knn40_graph, k=40, concat_points = True)

    #norm features 
    if points_normals_pl is not None:
        #normFeatures10 = normConv(points_normals_pl=points_normals_pl, knn_graph=knn10_graph, k=10, concat_points = False)  
        normFeatures20 = normConv(points_normals_pl=points_normals_pl, knn_graph=knn20_graph, k=20, concat_points = False)
        #normFeatures30 = normConv(points_normals_pl=points_normals_pl, knn_graph=knn30_graph, k=30, concat_points = False)
        normFeatures40 = normConv(points_normals_pl=points_normals_pl, knn_graph=knn40_graph, k=40, concat_points = True)
        #confusing code names but it is easy to write this way!!!
        edgeFeatures10 = normFeatures20
        edgeFeatures30 = normFeatures40
        print("edgeFeatures10", edgeFeatures10)
        print("edgeFeatures20", edgeFeatures20)
        print("edgeFeatures30", edgeFeatures30)
        print("edgeFeatures40", edgeFeatures40)

        
    #conv1 
    net = conv2d(inputs= edgeFeatures10,                 
                 num_output_channels=64,
                 kernel_size=[1,1],
                 scope='xyzcnn1',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)
    net =tf.reduce_max(net, axis=-2, keep_dims=True)
    net1 = net
    #print(net1)

    #conv2
    net = conv2d(inputs= edgeFeatures20,                 
                 num_output_channels=64,
                 kernel_size=[1,1],
                 scope='xyzcnn2',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net2 = net
    #print(net2)

    #conv3
    net = conv2d(inputs= edgeFeatures30,                 
                 num_output_channels=64,
                 kernel_size=[1,1],
                 scope='xyzcnn3',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)   
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net3 = net
    #print(net3)
    #conv4
    net = conv2d(inputs= edgeFeatures40,                 
                 num_output_channels=128,
                 kernel_size=[1,1],
                 scope='xyzcnn4',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)   
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
    net4 = net
    #print(net4)



    #agg
    net = conv2d(inputs= tf.concat([net1, net2, net3, net4], axis=-1),                 
                 num_output_channels=512,
                 kernel_size=[1,1],
                 scope='agg',
                 padding='VALID',
                 stride=[1,1], 
                 bn=True, 
                 decay=decay,
                 training=training)   
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
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
    #net = fully_connected(inputs = net,
    #                      num_outputs= 256,
    #                      scope = 'fc2',                          
    #                      bn=True,
    #                      decay= decay,
    #                      training = training)
    #print(net)

    #dp2
    #net = dropout(inputs=net, 
    #              training=training,
    #              scope='dp2',
    #              keep_prod=0.5)
    #print(net)

    #FC3
    net = fully_connected(inputs = net,
                          num_outputs= 40,
                          scope = 'fc3',                          
                          bn=True,
                          decay= decay,
                          training = training)
    #print(net)

    return net
def createModel(points_pl, training, knn_graph, k=20, decay=None):
    #input_points is a placeholder 
    batch_size = points_pl.get_shape()[0].value
    num_pts = points_pl.get_shape()[1].value       
    

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
    net =tf.reduce_max(net, axis=-2, keep_dims=True)
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
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
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
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
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
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
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
    net = tf.reduce_max(net, axis=-2, keep_dims=True)
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

    return net

############################################################################
###### Training  
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
def get_loss(pred, label):
    #pred: b*num_classes
    #labels: b
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss

def train_one_epoch(XYZ_point_cloud,                     
                    labels, 
                    sess, 
                    ops, 
                    train_writer,
                    XYZ_point_normals = None,
                    batch_size=32,
                    num_points=1024,
                    XYZ_point_notmals=None):                    

    #ops: dict mapping from string to tf ops
    is_training = True
    num_batches = XYZ_point_cloud.shape[0] // batch_size

    total_correct = 0
    total_seen = 0
    loss_sum = 0


    #for batch_idx in range(num_batches):
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size

        #get this batch data and labels
        current_data = XYZ_point_cloud[start_idx:end_idx,:,:]
        current_label = labels[start_idx:end_idx]
        if XYZ_point_notmals is not None:
            current_normals = XYZ_point_notmals[start_idx:end_idx,:,:]

        #sample random num_points of the data (does not affect the labels)        
        rand_ids = random.sample(range(XYZ_point_cloud.shape[1]), num_points)
        current_data = current_data[:,rand_ids,:]
        current_label = np.squeeze(current_label)
        if XYZ_point_notmals is not None:
            current_normals = current_normals[:,rand_ids,:]
            feed_dict = {ops['points_pl']:current_data,
                         ops['normals_pl']:current_normals,
                         ops['labels_pl']:current_label,
                         ops['is_training_pl']:is_training,}
        else:
            feed_dict = {ops['points_pl']:current_data,                         
                         ops['labels_pl']:current_label,
                         ops['is_training_pl']:is_training,}


        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], 
                                                         ops['step'],
                                                         ops['train_op'], 
                                                         ops['loss'], 
                                                         ops['pred']], 
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary,step)
        pred_val = np.argmax(pred_val,1)
        correct = np.sum(pred_val == current_label)
        total_correct += correct
        total_seen += batch_size
        loss_sum += loss_val
    log_string('mean loss :%f'%(loss_sum/ float(num_batches)))
    log_string('accuracy: %f'% (total_correct/float(total_seen)))

def eval_one_epoch(XYZ_point_cloud, 
                   labels,
                   sess,
                   ops,                 
                   batch_size=32,
                   num_points=1024,
                   num_classes=40,
                   XYZ_point_notmals=None):
    #ops is a dict mapping from string to ops
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]

    num_batches = XYZ_point_cloud.shape[0] // batch_size
    labels = np.squeeze(labels)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size

        #get this batch data and labels
        current_data = XYZ_point_cloud[start_idx:end_idx,:,:]        
        current_label = labels[start_idx:end_idx]
        
        rand_ids = random.sample(range(XYZ_point_cloud.shape[1]), num_points)
        current_data = current_data[:,rand_ids,:]
        if XYZ_point_notmals is not None:
            current_normals = XYZ_point_notmals[start_idx:end_idx,:,:]
            current_normals = current_normals[:,rand_ids,:]

            feed_dict = {ops['points_pl']: current_data,
                         ops['normals_pl']: current_normals,
                         ops['labels_pl']: current_label,
                         ops['is_training_pl']: is_training}
        else:
            feed_dict = {ops['points_pl']: current_data,                        
                         ops['labels_pl']: current_label,
                         ops['is_training_pl']: is_training}

        summary, step, loss_val, pred_val = sess.run([ops['merged'], 
                                                      ops['step'],
                                                      ops['loss'], 
                                                      ops['pred']], 
                                                     feed_dict=feed_dict)
        pred_val = np.argmax(pred_val,1)
        correct = np.sum(pred_val == current_label)
        total_correct += correct
        total_seen += batch_size
        loss_sum += (loss_val*batch_size)
        for i in range(start_idx, end_idx):
                l = labels[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


def trainMain(XYZ_point_cloud, labels, XYZ_point_notmals=None):
    
    print("Training")

    train_ids = list(range(0,9840))
    test_ids = list(range(9840,XYZ_point_cloud.shape[0]))

    #train_ids = list(range(0,10))
    #test_ids = list(range(10,20))

    batch_size = 32
    num_points = 1024
    k=20    

    pos_dim = 3
    max_epoch = 2000
    learning_rate = 0.001
    gpu = 0
    momentum = 0.9
    optimizer = 'adam'
    decay_step = 200000
    decay_rate = 0.7    
    num_classes = 40

    Dyn = True
    
    
    LOG_DIR = "log/"

    bn_init_decay = 0.5
    bn_decay_decay_rate = 0.5
    bn_decay_decay_step = float(decay_step)
    bn_decay_clip = 0.99

    with tf.Graph().as_default():       
        with tf.device('/device:GPU:'+str(gpu)):
            points_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, pos_dim))
            normals_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, pos_dim))
            labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
            
            if(Dyn):
            	knn10_graph =  KNN(pointcloud_pl=points_pl, k=10)
            	knn20_graph =  KNN(pointcloud_pl=points_pl, k=20)
            	knn30_graph =  KNN(pointcloud_pl=points_pl, k=30)
            	knn40_graph =  KNN(pointcloud_pl=points_pl, k=40)
            	#knn50_graph =  KNN(pointcloud_pl=points_pl, k=50)
            	#knn60_graph =  KNN(pointcloud_pl=points_pl, k=60)
            else:
            	knn_graph =  KNN(pointcloud_pl=points_pl, k=k)
            
            is_training_pl = tf.placeholder(tf.bool, shape=())
            

            batch = tf.Variable(0)
            bn_momentum = tf.train.exponential_decay(bn_init_decay,
                                                     batch*batch_size,
                                                     bn_decay_decay_step,
                                                     bn_decay_decay_rate,
                                                     staircase=True)
            bn_decay = tf.minimum(bn_decay_clip, 1 - bn_momentum)
            tf.summary.scalar('bn_decay', bn_decay)

            #create model, get loss
            if Dyn:
            	pred = createModel_Dyn(points_pl=points_pl, 
                                       points_normals_pl=normals_pl,
                	               	   training=is_training_pl,                     	               
                        	           knn10_graph=knn10_graph,
                        	           knn20_graph=knn20_graph,
                        	           knn30_graph=knn30_graph,
                        	           knn40_graph=knn40_graph,
                        	           #knn50_graph=knn50_graph,
                        	           #knn60_graph=knn60_graph,                        	           
                            	       decay=bn_decay)
            else:
            	pred = createModel(points_pl=points_pl, 
                	               training=is_training_pl, 
                    	           knn_graph=knn_graph,
                        	       k=k,
                            	   decay=bn_decay)
            loss = get_loss(pred=pred, label=labels_pl)

            tf.summary.scalar('loss',loss)

            correct = tf.equal(tf.argmax(pred,1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) /float(batch_size)
            tf.summary.scalar('accuracy', accuracy)

            #training op
            learning_rate = tf.train.exponential_decay(learning_rate,  
                                                       batch * batch_size,
                                                       decay_step,        
                                                       decay_rate,        
                                                       staircase=True)
            learning_rate = tf.maximum(learning_rate, 0.00001) 
            tf.summary.scalar('learning_rate', learning_rate)
            opt = tf.train.AdamOptimizer(learning_rate)
            train_op = opt.minimize(loss, global_step=batch)
                        
            saver = tf.train.Saver()

        #session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.allow_soft_placement = True        
        config.log_device_placement = False
        sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log/',sess.graph)
        

        #init var
        init = tf.global_variables_initializer()
        sess.run(init,{is_training_pl:True})

        ops ={'points_pl':points_pl,
              'normals_pl':normals_pl,
              'labels_pl':labels_pl,
              'is_training_pl': is_training_pl,
              'pred': pred,
              'loss':loss,
              'train_op': train_op,
              'merged':merged,
              'step':batch}
        for epoch in range(max_epoch):
            log_string('***** EPOCH %03d *****' %(epoch))
            sys.stdout.flush()

            train_one_epoch(XYZ_point_cloud = XYZ_point_cloud[train_ids,:,:], 
                            labels = labels[train_ids], 
                            sess = sess, 
                            ops = ops, 
                            train_writer=train_writer,
                            batch_size=batch_size,
                            num_points=num_points,
                            XYZ_point_notmals=XYZ_point_notmals[train_ids,:,:])
            
            eval_one_epoch(XYZ_point_cloud = XYZ_point_cloud[test_ids,:,:], 
                           labels = labels[test_ids],
                           sess = sess,
                           ops = ops,                          
                           batch_size=batch_size,
                           num_points=num_points,
                           num_classes=num_classes,
                           XYZ_point_notmals = XYZ_point_notmals[test_ids,:,:])

            if epoch %50 == 0:
              save_path = saver.save(sess,LOG_DIR+ "model.ckpt"+str (epoch))
              log_string("Model saved in file: %s" % save_path)    
############################################################################
###### Main 

if __name__ == "__main__":    

    if False:
        testModel()

    trainMain(XYZ_point_cloud, labels)