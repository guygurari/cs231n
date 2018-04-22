import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.
  
  D = num. features
  C = num. classes
  N = num. training samples

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Same for gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradient.
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # (implemented above)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  D = num. features
  C = num. classes
  N = num. training samples

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero, D x C

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W) # N x C
  correct_class_scores = np.choose(y, scores.T) # Basically X[i] . W[y[i]]
  margins = scores - correct_class_scores[:,np.newaxis] + 1 # N x C

  # These are all the ReLU(...) terms that sum to give the loss
  relus = np.maximum(np.zeros(margins.shape), margins)

  # Here we subtract the contribution from the y=y[i] terms, which should be
  # skipped over in the y sum, but for us they are equal to:
  #   max(0, correct_score - correct_score + 1) = 1
  # There are num_train such terms.
  loss += np.sum(relus) - num_train

  loss /= num_train

  # Regularization sum
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  step_function = np.piecewise(margins, [margins<0, margins>=0], [0, 1]) # N x C

  # Add the first loss sum
  dW += X.T.dot(step_function)

  step_function_summed_over_classes = step_function.sum(axis=1) # N

  all_classes = np.zeros((num_train, num_classes)) # N x C
  all_classes += np.arange(num_classes)

  # N x C
  # Equals to: \delta_{y[i], y'}
  # For each sample, contains 1 in the correct class and 0 otherwise
  correct_class_indicator = (all_classes == y[:,np.newaxis]).astype(int) # N x C

  # N x C
  indicator_times_step_func = correct_class_indicator * \
                              step_function_summed_over_classes[:,np.newaxis]

  # Add the second loss sum
  dW -= X.T.dot(indicator_times_step_func)
  
  # Normalize
  dW /= num_train

  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
