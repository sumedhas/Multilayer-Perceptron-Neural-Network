'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle

from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import time


# Do not change this
def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W


# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-z))

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    # 1-of-K coding scheme
    oneOfK = np.zeros((training_label.shape[0], 2))
    oneOfK[np.arange(training_label.shape[0], dtype="int"), training_label.astype(int)] = 1
    training_label = oneOfK

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    # Forward Propagation
    a = np.array(training_data)
    b = np.array(np.ones(training_data.shape[0]))

    # Bias
    training_data = np.column_stack((a, b))

    # Feed forward pass from input to hidden layer
    # Dot product of two arrays. rever to 3.2.3 (2)
    zj = sigmoid(np.dot(training_data, w1.T))

    # Bias
    zj = np.column_stack((zj, np.ones(zj.shape[0])))
    # Feed from hidden to output layer
    ol = sigmoid(np.dot(zj, w2.T))

    # Error function and Back propogation
    # Delta=(y-yhat)
    delta = ol - training_label

    # Gradient decent w2
    w2_gradient = np.dot(delta.T, zj)

    # Using formula from (11 & 12)
    w1_gradient = np.dot(((1 - zj) * zj * (np.dot(delta, w2))).T, training_data)

    # Remove zero row
    w1_gradient = np.delete(w1_gradient, n_hidden, 0)

    # calculating obj_val
    n = training_data.shape[0]
    error = (np.sum(-1 * (training_label * np.log(ol) + (1 - training_label) * np.log(1 - ol)))) / n
    obj_val = error + ((lambdaval / (2 * n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))

    # Regularization in Neural Network - return obj_value and obj_gradient
    # obj val
    # obj grad

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array

    # Partial derivative of new objective function with respect to weight hidden to output layer
    w2_gradient = (w2_gradient + (lambdaval * w2)) / n
    # Partial derivative of new objective function with respect to weight input to hidden layer
    w1_gradient = (w1_gradient + (lambdaval * w1)) / n

    # calculating obj_grad
    obj_grad = np.array([])
    obj_grad = np.concatenate((w1_gradient.flatten(), w2_gradient.flatten()), 0)

    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    labels = np.array([])
    # Your code here
    # adding bias node in input vector
    bias = np.ones(len(data))
    data = np.column_stack([data, bias])
    # compute data*wtrans for input to hidden
    ans1 = data.dot(w1.T)
    # compute sigmoid of data*wtrans
    sig1 = sigmoid(ans1)
    # adding bias node in hidden vector
    sig_bias = np.ones(len(data))
    # sig_bias = np.ones(sig1.shape[0])
    sig1 = np.column_stack([sig1, sig_bias])
    # compute sig1*wtrans for hidden to output
    ol = sig1.dot(w2.T)
    # compute sigmoid of sig1*wtrans
    sig2 = sigmoid(ol)
    # selecting max value for predicted label
    '''
    for each in range(sig2.shape[0]):
        labels[each] = np.argmax(sig2[each])
    '''
    labels = np.argmax(sig2, axis=1)

    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


"""**************Neural Network Script Starts here********************************"""

timer = time.time()

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
params = nn_params.get('x')
# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data)
# find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, validation_data)
# find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, test_data)
# find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

getTime = time.time()-timer
print('\n It took: ' + str(getTime)+ 'seconds to complete')
