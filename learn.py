import tensorflow as tf
import cv2
import pong             #our class written in pong.py
import numpy as np 
import random
from collections import deque #queue data structure for fast appends pops


# We're going to make a 5 layer convo neural network
# read image data + score
# spit out function


ACTIONS = 3             # paddle actions (up, down, stay)

GAMMA = .99             # learning rate

INITIAL_EPSILON = 1.0   # for updating our gradient/training over time
                        # when we update training, we want to decrease epsilon
FINAL_EPSILON = 0.05 

EXPLORE = 500000        # how many frames to anneal epsilon 

OBSERVE = 50000

REPLAY_MEMORY = 50000   # store our experiences the size of it

BATCH = 100             # batch size to train on, ie how many times we want to train

# create tensorflow gradient 
def createGraph():
    
    # convolutional layers
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64])) 
    
    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    # input for pixel data 
    s = tf.placeholder("float", [None, 84, 84, 4])

    # computes rectified linear unit activiation function on a 2D
    # convolution given 4d input and filter tensors. and
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1],
        padding = "VALID") + b_conv1)

    conv2 = tf.nn.relu(tf.nn.conv2d(s, W_conv2, strides = [1, 2, 2, 1],
        padding = "VALID") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(s, W_conv3, strides = [1, 1, 1, 1],
        padding = "VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    tc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5

def trainGraph(inp, out, sess):

    #to calculate the argmax, we multiply the predicted output with a vector
    # with oone value  and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None]) #ground truth

    #action 
    action = tf.reduce_sum(tf.mul(out))



