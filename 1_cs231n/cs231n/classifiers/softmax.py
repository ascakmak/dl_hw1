import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
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
  N = X.shape[1] # N is number of Samples
  C = W.shape[0] # C is number of classes
  f = np.matmul(W,X)
  f -= np.amax(f, axis=0) # Shifts all elements of each column so that max is 0
  p = np.exp(f) / np.sum(np.exp(f), axis=0) # C x N (p is the softmax probs)
  reg_loss = 0.5*reg*np.sum(W * W) # Regularization Loss
  loss = np.mean(-np.log(p[y, np.arange(N)] + 1e-10)) + reg_loss # Total Loss
  dscores = np.copy(p)
  dscores[y,range(N)] -= 1
  dscores /= N  # C x N
  dW = np.dot(dscores, X.T) # dscores is C x N X.T is N x D
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
