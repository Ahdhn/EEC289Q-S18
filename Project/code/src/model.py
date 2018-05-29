import tensorflow as tf
import numpy as np
import math

if __name__ == '__main__':
    #use this for testing only
    #for actual training, see train.py

    batch_size=2
    num_pts = 124
    dim=3

    #generate random points, assigne labels to them as 1 and 0
    input_feed = np.random.rand(batch_size, num_pts, dim)
    label_feed = np.random(rand, batch_size)
    label_feed[label_feed >=0.5]=1
    label_feed[label_feed <0.5]=0
    #cast to int 
    label_feed = label_feed.astype(np.int) 

    #with tf.Graph().as_default():#not needed but it is a good practice 
    #    #placeholder for the points