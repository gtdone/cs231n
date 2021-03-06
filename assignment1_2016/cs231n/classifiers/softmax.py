import numpy as np
from random import shuffle

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
  num_class = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
        scores= X[i].dot(W)
        normalized_scores = np.exp(scores)/np.sum(np.exp(scores))
        loss += -np.log(normalized_scores[y[i]])
         
        dscore = np.reshape(normalized_scores, (num_class,1))*X[i]
        dscore[y[i],:] -=X[i]
        dW += dscore.T
  
  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW = dW/num_train + reg*W  
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
  num_train= X.shape[0]

  scores = X.dot(W)
  normalized_scores = np.exp(scores).T/np.sum(np.exp(scores).T, axis=0)
  
  loss= np.sum( -np.log(normalized_scores[y, range(num_train)]))
  loss= loss / num_train + 0.5*reg*np.sum(W*W)
  
  dscore = normalized_scores
  dscore[y, range(num_train)] -=1
  dW = X.T.dot(dscore.T) / num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

