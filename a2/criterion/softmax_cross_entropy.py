import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11


class SoftmaxCrossEntropyLossLayer:
    def __init__(self):
        self.acc = 0.0
        self.loss = np.zeros(1, dtype="f")

    def forward(self, logit, gt):
        """
        Inputs: (minibatch)
        - logit: forward results from the last FCLayer, shape(batch_size, 10)
        - gt: the ground truth label, shape(batch_size, 10)
        """

        ############################################################################
        # TODO: Put your code here
        # Calculate the average accuracy and loss over the minibatch, and
        # store in self.accu and self.loss respectively.
        # Only return the self.loss, self.accu will be used in solver.py.
        self.logit = logit
        self.gt = gt

        shifted_logits = logit - np.max(logit, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        self.prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        log_likelihood = -np.log(np.sum(self.prob * gt, axis=1) + EPS)
        self.loss = np.mean(log_likelihood)

        pred = np.argmax(self.prob, axis=1)
        truth = np.argmax(gt, axis=1)
        self.acc = np.mean(pred == truth)
        ############################################################################

        return self.loss

    def backward(self):
        ############################################################################
        # TODO: Put your code here
        # Calculate and return the gradient (have the same shape as logit)
        batch_size = self.logit.shape[0]
        grad = (self.prob - self.gt) / batch_size
        return grad
        ############################################################################
