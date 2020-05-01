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
  dW_i = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    temp_score = np.dot(X[i],W)
    correct_class_score = temp_score[y[i]]
    denominator = np.sum(np.exp(temp_score))
    soft_prob = np.exp(correct_class_score)/denominator
    loss_i = -np.log(soft_prob)
    loss+=loss_i
    #dW_i = soft_prob * X[i] - y[i] *X[i]
    
    #dW += dW_i
    
    for j in range(num_classes):
      nume = np.exp(temp_score[j])
      dW_i[:,j] = nume/denominator * X[i]
      if j == y[i]:
        dW_i[:,y[i]] = (nume/denominator - 1) * X[i]
    dW +=dW_i
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
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
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  temp_score = np.dot(X,W)
  denominator = np.sum(np.exp(temp_score), axis =1, keepdims = True)
  temp_soft = np.exp(temp_score)/denominator
  temp_loss = np.sum(-np.log(temp_soft[range(num_train), y]))
  loss = (temp_loss/num_train) + (reg * np.sum(W * W))

  temp_soft[range(num_train), y] -= 1
  temp_soft /= num_train
  dW = np.dot(X.T, temp_soft) + reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

