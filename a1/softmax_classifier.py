import numpy as np


def softmax_classifier(W, input, label, _lambda):
    """
    Softmax Classifier

    Inputs have dimension D, there are C classes, a minibatch have N examples.
    (In this homework, D = 784, C = 10)

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - input: A numpy array of shape (N, D) containing a minibatch of data.
    - label: A numpy array of shape (N, C) containing labels, label[i] is a
      one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
    - _lambda: regularization strength, which is a hyerparameter.

    Returns:
    - loss: a single float number represents the average loss over the minibatch.
    - gradient: shape (D, C), represents the gradient with respect to weights W.
    - prediction: shape (N,), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here

    # generate the current prediction matrix
    Weights = input.dot(W)  # (N, C)
    Weights_max = np.max(Weights, axis=1, keepdims=True)
    Weights_diff = Weights - Weights_max
    exp_Weights = np.exp(Weights_diff)
    probabilities = exp_Weights / np.sum(exp_Weights, axis=1, keepdims=True)

    # calculate the loss given the result matrix probabilities
    training_size = input.shape[0]
    loss = -np.sum(label * np.log(probabilities)) / training_size
    loss += 0.5 * _lambda * np.sum(W * W)

    # then we can calculate the gradient given the loss
    gradient = input.T.dot(probabilities - label) / training_size
    gradient += _lambda * W

    # finally return the max value index of each row in probabilities as prediction
    prediction = np.argmax(probabilities, axis=1)

    ############################################################################

    return loss, gradient, prediction
