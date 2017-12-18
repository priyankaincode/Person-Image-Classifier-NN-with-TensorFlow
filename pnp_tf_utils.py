import h5py
import numpy as np
import tensorflow as tf
import math

def load_data():

    ### Loads the image data (train, validation, test) in from an HDF5 file

    tvt_dataset = h5py.File('Dataset/dataset.hdf5', "r")
    train_set_x_orig = np.array(tvt_dataset["train_img"][:]) # train set features
    train_set_y_orig = np.array(tvt_dataset["train_labels"][:]) # train set labels

    val_set_x_orig = np.array(tvt_dataset["val_img"][:]) # validation set features
    val_set_y_orig = np.array(tvt_dataset["val_labels"][:]) # validation set labels

    test_set_x_orig = np.array(tvt_dataset["test_img"][:]) # test set features
    test_set_y_orig = np.array(tvt_dataset["test_labels"][:]) # test set labels
    
    return train_set_x_orig, train_set_y_orig, val_set_x_orig, val_set_y_orig, test_set_x_orig, test_set_y_orig


def create_placeholders(n_x, n_y):

    ### Creates the placeholders for the tensorflow session.
    
    ###  Arguments:
    ###  n_x -- Number of inputs
    ###  n_y -- Number of outputs
    
    ###  Returns:
    ###  X -- Placeholder for the data input, of shape [n_x, None] and dtype "float63"
    ###  Y -- Placeholder for the input labels, of flexible shape and dtype "float63"

    X = tf.placeholder(tf.float64, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float64, name="Y")
    
    return X, Y


def initialize_parameters():
   
    ### Initializes parameters to build a neural network with tensorflow. The shapes are:
    ###   W1 : [30, 230400]
    ###   b1 : [30, 1]
    ###   W2 : [6, 30]
    ###   b2 : [6, 1]
    ###   W3 : [4, 6]
    ###   b3 : [4, 1]
    ###   W4 : [1, 4]
    ###   b4 : [1, 1]
    
    ### Returns:
    ### parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3, W4, b4

    W1 = tf.get_variable("W1", [30,230400], dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [30,1], dtype=tf.float64, initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [6,30], dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [6,1], dtype=tf.float64, initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [4,6], dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [4,1], dtype=tf.float64, initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [1,4], dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [1,1], dtype=tf.float64, initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}
    
    return parameters


def forward_propagation(X, parameters):

    ### Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    ###  Arguments:
    ###  X -- Input dataset placeholder, of shape (input size, number of examples)
    ###  parameters -- Python dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"

    ###  Returns:
    ###  Z3 -- The output of the last linear unit
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    # Forward propagation
    Z1 = tf.add(tf.matmul(W1, X), b1)                                             
    A1 = tf.nn.relu(Z1)                                              
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              
    A2 = tf.nn.relu(Z2)                         
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    return Z4

def compute_cost(Z4, Y, parameters, lambd, m):

    ### Computes the cost
    
    ###  Arguments:
    ###  Z4 -- Output of forward propagation (output of the last linear unit)
    ###  Y -- "True" labels vector placeholder, same shape as Z4
    
    ###  Returns:
    ###  cost - Tensor of the cost function

    
    logits = tf.transpose(Z4)
    labels = Y
    
    # Calculate unregularized cost and regularization factor and then add them together (L2 regularization method)
    cost_unreg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.cast(labels, tf.float64)))
    reg = (lambd/(2*m)) * (tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) + tf.nn.l2_loss(parameters["W3"]) + tf.nn.l2_loss(parameters["W4"]))

    cost = tf.add(cost_unreg, reg)
    
    return cost

def random_mini_batches(X, Y, mini_batch_size):

    ### Creates a list of random minibatches from (X, Y)
    
    ###  Arguments:
    ###  X -- Input data, of shape (input size, number of examples)
    ###  Y -- True "label" vector 
    ###  mini_batch_size -- Size of the mini-batches, integer
    
    ###  Returns:
    ###  mini_batches -- List of synchronous (mini_batch_X, mini_batch_Y)
   
    
    m = X.shape[1]                 
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y), minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k: mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[mini_batch_size * k: mini_batch_size * (k+1)]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size: m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size: m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
