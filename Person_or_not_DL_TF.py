##### THIS LEARNING ALGORITHM DECIDES WHETHER AN IMAGE CONTAINS A PERSON OR NOT - BUT THIS TIME, USING TENSORFLOW
##### INPUT: IMAGE 
##### OUTPUT: Y/N AS TO WHETHER IT CONTAINS A PERSON


import math
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from pnp_tf_utils import *

# Load dataset
train_x_orig, train_y_orig, val_x_orig, val_y_orig, test_x_orig, test_y_orig = load_data()

m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
m_val = val_x_orig.shape[0]

num_px_x = train_x_orig.shape[1]
num_px_y = train_x_orig.shape[2]

# Reshape labels for ease of use
train_y = train_y_orig.reshape(len(train_y_orig), 1)
val_y = val_y_orig.reshape(len(val_y_orig), 1)
test_y = test_y_orig.reshape(len(test_y_orig), 1)

# Flatten train and val and test images
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  
val_x_flatten = val_x_orig.reshape(val_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
val_x = val_x_flatten/255.
test_x = test_x_flatten/255.


# Print some facts about the dataset
print("Some information about this dataset: ")
print ("Number of training examples: " + str(m_train))
print ("Number of validation examples: " + str(m_val))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px_x) + ", " + str(num_px_y) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("val_x_orig shape: " + str(val_x_orig.shape))
print ("val_y shape: " + str(val_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

print ("train_x's flattened shape: " + str(train_x.shape))
print ("val_x's flattened shape: " + str(val_x.shape))
print ("test_x's flattened shape: " + str(test_x.shape))

input("\nPress Enter to continue.")


# Define model, train it, and run it on train/val/test sets
def model(X_train, Y_train, X_val, Y_val, X_test, Y_test, learning_rate, num_epochs, lambd, print_cost):
    
    ### Implements a five-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    ###  Arguments:
    ###  X_train -- Training set, of shape (input size, number of training examples)
    ###  Y_train -- Test set, of shape (output size, number of training examples)
    ###  X_test -- Training set, of shape (input size, number of training examples)
    ###  Y_test -- Test set, of shape (output size, number of test examples)
    ###  learning_rate -- Learning rate of the optimization
    ###  num_epochs -- Number of epochs of the optimization loop
    ###  print_cost -- True to print the cost every 100 epochs

    ###  Returns:
    ###  parameters -- Parameters learned by the model
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y) (input size, output size)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    z4 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost1 = compute_cost(z4, Y, parameters, lambd, m)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost1)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
                
            # Run the session to execute the optimizer and the cost 
            _ , cost = sess.run([optimizer, cost1], feed_dict={X: X_train, Y: Y_train})

                
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(cost)
                
        # Accuracy measure (compares rounded predictions to Y values)
        correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(z4)), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Print the accuracies
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Validation Accuracy:", accuracy.eval({X: X_val, Y: Y_val}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

# MODEL WITH MINIBATCHES. NOTE: THIS MODEL TAKES AN EXCESSIVELY LONG TIME TO TRAIN WITH THIS DATASET AND IS THUS NOT CALLED AT ANY TIME
def model_with_minibatches(X_train, Y_train, X_test, Y_test, learning_rate, minibatch_size, num_epochs, lambd, print_cost):
    
    ### Implements a five-layer tensorflow neural network using mini-batches: LINEAR->RELU->LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    ###  Arguments:
    ###  X_train -- Training set, of shape (input size, number of training examples)
    ###  Y_train -- Test set, of shape (output size, number of training examples)
    ###  X_test -- Training set, of shape (input size, number of training examples)
    ###  Y_test -- Test set, of shape (output size, number of test examples)
    ###  learning_rate -- Learning rate of the optimization
    ###  minibatch_size -- Size of a minibatch
    ###  num_epochs -- Number of epochs of the optimization loop
    ###  print_cost -- True to print the cost every 100 epochs

    ###  Returns:
    ###  parameters -- Parameters learned by the model
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y) (input size, output size)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    z4 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(z4, Y, parameters, lambd, m)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) 
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(minibatch_cost)
                
        # Accuracy measure (compares rounded predictions to Y values)
        correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(z4)), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained! Now let's see how accurate it is...")	

        
        return parameters


# Run model
parameters = model(train_x, train_y, val_x, val_y, test_x, test_y, learning_rate = 0.01, num_epochs = 1500, lambd = 0.8, print_cost = True)


