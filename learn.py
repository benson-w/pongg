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

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides = [1, 2, 2, 1],
        padding = "VALID") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1],
        padding = "VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5

def trainGraph(inp, out, sess):

    #to calculate the argmax, we multiply the predicted output with a vector
    # with oone value  and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None]) #ground truth

    #action 
    action = tf.reduce_sum(tf.mul(out, argmax), reduction_indices = 1)
    #cost function we will reduce through back propagation
    cost = tf.reduce_mean(tf.square(action - gt))
    #optimization function to reduce our cost fn
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #initialize our game
    game = pong.PongGame()

    #create a queue for experience replay to store policies 
    D = deque()

    #initial frame
    frame = game.getPresentFrame()
    #convert rgb to gray scale for processing
    # also reduce the size from 400, 400 to 84, 84
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    #binary colors, black/white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    #stack frames, that is our input tensor 
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    #saver 
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON

    #training time
    while(1):
        #output tensor       
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
        #argmax function
        argmax_t = np.zeros([ACTIONS])

        if (random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE 

        #reward tensor if score is positive
        reward_t, frame = game.getNextFrame(argmax_t)
        #get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        #new input tensor 
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)
        
        #add our input tensor argmax tensor, reward and updated input tensor
        # to stack of experiences 

        D.append((inp_t, argmax_t, reward_t, inp_t1))
        #if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #training iteration
        if t > OBSERVE:
            #get values from our replay memory
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict = {inp : inp_t1_batch})

            #add values to our batch 
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            #train on this
            train_step.run(feed_dict = {
                           gt : gt_batch,
                           argmax : argmax_batch,
                           inp : inp_batch 
                           })
            
        #update our input tensor with the next frame
        inp_t = inp_t1
        t = t + 1

        
        
        #print out where we are after saving where we are
        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step = t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex,
            "/REWARD", reward_t, "/Q_MAX %e" % np.max(out_t))

def main():
    #create session
    sess = tf.InteractiveSession()
    #input layer and output layer by creating graph
    inp, out = createGraph()
    #train our graph on input and output with session variables
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()
        
