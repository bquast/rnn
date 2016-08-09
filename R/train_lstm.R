#' @name train_lstm
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid sigmoid_output_to_derivative tanh_output_to_derivative
#' @title Long Short-Term Memory RNN
#' @description Trains a Long Short-Term Memory RNN
#' @param Y array of output values, dim 1: samples (must be equal to dim 1 of X), dim 2: time (must be equal to dim 2 of X), dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param learningrate learning rate to be applied for weight iteration
#' @param numepochs number of iteration, i.e. number of time the whole dataset is presented to the network
#' @param hidden_dim dimension(s) of hidden layer(s)
#' @param learningrate_decay coefficient to apply to the learning rate at each weight iteration
#' @param momentum coefficient of the last weight iteration to keep for faster learning
#' @param use_bias should the network use bias
#' @return a model to be used by the predict_lstm function
#' @examples 
#' # example for LSTM
#'
#' ## convert integer to binary
#' i2b <- function(integer, length=8)
#'   as.numeric(intToBits(integer))[1:length]
#' 
#' ## apply 
#' int2bin <- function(integer, length=8)
#'   t(sapply(integer, i2b, length=length))
#' 
#' # create training numbers
#' X1 = sample(0:1023, 100000, replace=TRUE)
#' X2 = sample(0:1023, 100000, replace=TRUE)
#'
#' # create training response numbers
#' Y <- X1 + X2
#'
#' # convert to binary
#' X1b <- int2bin(X1, length=10)
#' X2b <- int2bin(X2, length=10)
#' Yb  <- int2bin(Y,  length=10)
#'

train_lstm <- function(Y, X, learningrate, learningrate_decay = 1, momentum = 0, hidden_dim = c(10), numepochs = 1, use_bias = FALSE) {
  # all the things
  
  # use predict_lstm function here
  
  # backprop
  
  # output model
}