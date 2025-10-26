import numpy as np


class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.

    def forward(self, logit, gt):
        """
      Inputs: (minibatch)
      - logit: forward results from the last FCLayer, shape(batch_size, 10)
      - gt: the ground truth label, shape(batch_size, 10)
    """

        ############################################################################
    # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.acc and self.loss respectively.
        # Only return the self.loss, self.acc will be used in solver.py.
        batch_size = logit.shape[0]
        self.loss = (np.sum((logit - gt) ** 2) / 2) / batch_size

        prediction = np.argmax(logit, axis=1)
        truth = np.argmax(gt, axis=1)

        self.acc = np.sum(prediction == truth) / batch_size

        self.logit = logit
        self.gt = gt
    ############################################################################

        return self.loss

    def backward(self):

        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        grad = (self.logit - self.gt) / self.logit.shape[0]
        return grad
        ############################################################################
