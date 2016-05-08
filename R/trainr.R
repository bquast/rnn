#' @name trainr
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid logistic sigmoid_output_to_derivative
#' @title Recurrent Neural Network
#' @description Trains a Recurrent Neural Network.
#' @param Y array of output values, dim 1: samples (must be equal to dim 1 of X), dim 2: time (must be equal to dim 2 of X), dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param learningrate learning rate to be applied for weight iteration
#' @param numepochs number of iteration, i.e. number of time the whole dataset is presented to the network
#' @param hidden_dim dimension of hidden layer
#' @param start_from_end should the sequence start from the end
#' @param learningrate_decay coefficient to apply to the learning rate at each weight iteration
#' @param momentum coefficient of the last weight iteration to keep for faster learning
#' @return a model to be used by the predictr function
#' @examples 
#' # create training numbers
#' X1 = sample(0:127, 7000, replace=TRUE)
#' X2 = sample(0:127, 7000, replace=TRUE)
#' 
#' # create training response numbers
#' Y <- X1 + X2
#' 
#' # convert to binary
#' X1 <- int2bin(X1, length=8)
#' X2 <- int2bin(X2, length=8)
#' Y  <- int2bin(Y,  length=8)
#' 
#' # create 3d array: dim 1: samples; dim 2: time; dim 3: variables
#' X <- array( c(X1,X2), dim=c(dim(X1),2) )
#' 
#' # train the model
#' model <- trainr(Y=Y,
#'                 X=X,
#'                 learningrate   =  0.1,
#'                 hidden_dim     = 10,
#'                 start_from_end = TRUE )
#'     

trainr <- function(Y, X, learningrate, learningrate_decay = 1, momentum = 0, hidden_dim, numepochs = 1, start_from_end=FALSE) {
  
  # check the consistency
  if(dim(X)[2] != dim(Y)[2]){
    stop("The time dimension of X is different from the time dimension of Y. Only sequences to sequences is supported")
  }
  if(dim(X)[1] != dim(Y)[1]){
    stop("The sample dimension of X is different from the sample dimension of Y.")
  }
  
  # coerce to array if matrix
  if(length(dim(X)) == 2){
    X <- array(X,dim=c(dim(X),1))
  }
  if(length(dim(Y)) == 2){
    Y <- array(Y,dim=c(dim(Y),1))
  }
  
  # extract the network dimensions
  input_dim = dim(X)[3]
  output_dim = dim(Y)[3]
  binary_dim = dim(X)[2]
  
  # initialize neural network weights
  synapse_0 = matrix(stats::runif(n = input_dim*hidden_dim, min=-1, max=1), nrow=input_dim)
  synapse_1 = matrix(stats::runif(n = hidden_dim*output_dim, min=-1, max=1), nrow=hidden_dim)
  synapse_h = matrix(stats::runif(n = hidden_dim*hidden_dim, min=-1, max=1), nrow=hidden_dim)
  
  # initialize the update
  synapse_0_update = matrix(0, nrow = input_dim, ncol = hidden_dim)
  synapse_1_update = matrix(0, nrow = hidden_dim, ncol = output_dim)
  synapse_h_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)
  
  # initialize the old update for the momentum
  synapse_0_old_update = matrix(0, nrow = input_dim, ncol = hidden_dim)
  synapse_1_old_update = matrix(0, nrow = hidden_dim, ncol = output_dim)
  synapse_h_old_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)

  
  # Storing layers states
  store_output <- array(0,dim = dim(Y))
  store_hidden <- array(0,dim = c(dim(Y)[1:2],hidden_dim))
  
  # Storing errors, dim 1: samples, dim 2 is epochs, we could store also the time and variable dimension
  error <- array(0,dim = c(dim(Y)[1],numepochs))
  
  # training logic
  for(epoch in seq(numepochs)){
    message(paste0("Training epoch: ",epoch," - Learning rate: ",learningrate))
    for (j in 1:dim(Y)[1]) {
      
      # generate a simple addition problem (a + b = c)
      a = array(X[j,,],dim=c(dim(X)[2],input_dim))
      
      # true answer
      c = array(Y[j,,],dim=c(dim(Y)[2],output_dim))
      
      overallError = 0
      
      layer_2_deltas = matrix(0,nrow=1, ncol = output_dim)
      layer_1_values = matrix(0, nrow=1, ncol = hidden_dim)
      # layer_1_values = rbind(layer_1_values, matrix(0, nrow=1, ncol=hidden_dim))
      
      # time index vector, needed because we predict in one direction but update the weight in an other
      if(start_from_end == TRUE) {
        pos_vec      <- binary_dim:1
        pos_vec_back <- 1:binary_dim
      } else {
        pos_vec      <- 1:binary_dim
        pos_vec_back <- binary_dim:1
      }
      
      # moving along the time
      for (position in pos_vec) {
        
        # generate input and output
        x = a[position,]
        y = c[position,]
        
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid::logistic((x%*%synapse_0) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h))
        
        # output layer (new binary representation)
        layer_2 = sigmoid::logistic(layer_1 %*% synapse_1)
        
        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas = rbind(layer_2_deltas, layer_2_error * sigmoid::sigmoid_output_to_derivative(layer_2))
        overallError = overallError + sum(abs(layer_2_error))
        
        # storing
        store_output[j,position,] = layer_2
        store_hidden[j,position,] = layer_1
        
        # store hidden layer so we can print it out. Needed for error calculation and weight iteration
        layer_1_values = rbind(layer_1_values, layer_1)
        
      }
      
      # store errors
      error[j,epoch] <- overallError
      
      future_layer_1_delta = matrix(0, nrow = 1, ncol = hidden_dim)
      
      # Weight iteration,
      for (position in 0:(binary_dim-1)) {
        
        x            = a[pos_vec_back[position+1],]
        layer_1      = layer_1_values[dim(layer_1_values)[1]-position,]
        prev_layer_1 = layer_1_values[dim(layer_1_values)[1]-(position+1),]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[dim(layer_2_deltas)[1]-position,]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta %*% t(synapse_h) + layer_2_delta %*% t(synapse_1)) *
          sigmoid::sigmoid_output_to_derivative(layer_1)
        
        # let's update all our weights so we can try again
        synapse_1_update = synapse_1_update + matrix(layer_1) %*% layer_2_delta
        synapse_h_update = synapse_h_update + matrix(prev_layer_1) %*% layer_1_delta
        synapse_0_update = synapse_0_update + c(x) %*% layer_1_delta # I had to change X as a vector as it is not a matrix anymore, other option, define it as a matrix of dim()=c(1,input_dim)
        
        future_layer_1_delta = layer_1_delta
      }
      
      # Calculate the real update including learning rate and momentum
      synapse_0_update = synapse_0_update * learningrate + synapse_0_old_update * momentum
      synapse_1_update = synapse_1_update * learningrate + synapse_1_old_update * momentum
      synapse_h_update = synapse_h_update * learningrate + synapse_h_old_update * momentum
      
      # Applying the update
      synapse_0 = synapse_0 + synapse_0_update
      synapse_1 = synapse_1 + synapse_1_update
      synapse_h = synapse_h + synapse_h_update
      
      # Update the learning rate
      learningrate <- learningrate * learningrate_decay
      
      # Storing the old update for next momentum
      synapse_0_old_update = synapse_0_update
      synapse_1_old_update = synapse_1_update
      synapse_h_old_update = synapse_h_update
      
      # Initializing the update
      synapse_0_update = synapse_0_update * 0
      synapse_1_update = synapse_1_update * 0
      synapse_h_update = synapse_h_update * 0
    }
    # update best guess if error is minimal
    if(colMeans(error)[epoch] <= min(colMeans(error)[1:epoch])){
      store_output_best <- store_output
      store_hidden_best <- store_hidden
    }
    message(paste0("Epoch error: ",colMeans(error)[epoch]))
  }
  
  
  # output object with synapses
  return(list(synapse_0         = synapse_0,
              synapse_1         = synapse_1,
              synapse_h         = synapse_h,
              error             = error,
              store_output      = store_output,
              store_hidden      = store_hidden,
              store_hidden_best = store_hidden_best,
              store_output_best = store_output_best,
              start_from_end    = start_from_end) )
  
}
