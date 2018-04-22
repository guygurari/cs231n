import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]

  exp_scores = np.exp(X.dot(W)) # (N, C)

  for i in xrange(num_train):
    denom = 0.
    for j in xrange(num_classes):
        denom += exp_scores[i,j]
    loss += - np.log( exp_scores[i, y[i]]  / denom )

    for j in xrange(num_classes):
      for k in xrange(num_features):
        factor = exp_scores[i, j] / denom
        if j == y[i]:
          factor -= 1
        dW[k,j] += X[i,k] * factor

  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]

  exp_scores = np.exp(X.dot(W)) # (N, C)
  exp_scores_summed_over_classes = exp_scores.sum(axis=1) # (N)
  correct_class_exp_scores = np.choose(y, exp_scores.T) # (N)
  loss = - np.sum(np.log(correct_class_exp_scores / exp_scores_summed_over_classes))
  loss /= num_train

  # (N, C)
  exps_fraction = exp_scores / exp_scores_summed_over_classes[:,np.newaxis]

  all_classes = np.zeros((num_train, num_classes)) # (N, C)
  all_classes += np.arange(num_classes)
  # N x C
  # Equals to: \delta_{y[i], y'}
  # For each sample, contains 1 in the correct class and 0 otherwise
  correct_class_indicator = (all_classes == y[:,np.newaxis]).astype(int) # (N, C)

  dW = X.T.dot(exps_fraction - correct_class_indicator) / num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

