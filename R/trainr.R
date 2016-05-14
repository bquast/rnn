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
#' @param hidden_dim dimension(s) of hidden layer(s)
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

trainr <- function(Y, X, learningrate, learningrate_decay = 1, momentum = 0, hidden_dim = c(10), numepochs = 1, start_from_end=FALSE) {
  
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
  synapse_dim = c(input_dim,hidden_dim,output_dim)
  binary_dim = dim(X)[2]
  
  #initialize neural network weights, stored in two lists
  time_synapse      = list() # synapse in a time step, link input to hidden, hidden to hidden, hidden to output
  recurrent_synapse = list() # synapse between time step, link hidden to hidden
  for(i in seq(length(synapse_dim) - 1)){
    time_synapse[[i]] <- matrix(runif(n = synapse_dim[i]*synapse_dim[i+1], min=-1, max=1), nrow=synapse_dim[i])
  }
  for(i in seq(length(hidden_dim))){
    recurrent_synapse[[i]] <- matrix(runif(n = hidden_dim[i]*hidden_dim[i], min=-1, max=1), nrow=hidden_dim[i])
  }
  
  # initialize the update, stored in two lists
  time_synapse_update = time_synapse
  time_synapse_update = lapply(time_synapse_update,function(x){x*0})
  recurrent_synapse_update = recurrent_synapse
  recurrent_synapse_update = lapply(recurrent_synapse_update,function(x){x*0})
  
  # Storing layers states, filled with 0 for the moment
  store <- list()
  for(i in seq(length(synapse_dim) - 1)){
    store[[i]] <- array(0,dim = c(dim(Y)[1:2],synapse_dim[i+1]))
  }
  
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
      # store the hidden layers values for each time step, needed in parallel of store because we need the t(-1) hidden states. otherwise, we could take the values from the store list
      layers_values  = list()
      for(i in seq(length(synapse_dim) - 2)){
        layers_values[[i]] <- matrix(0,nrow=1, ncol = synapse_dim[i+1])
      }
      
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
        
        layers <- list()
        for(i in seq(length(synapse_dim) - 1)){
          if (i == 1) { # first hidden layer, need to take x as input
            layers[[i]] <- logistic((x%*%time_synapse[[i]]) + (layers_values[[i]][dim(layers_values[[i]])[1],] %*% recurrent_synapse[[i]]))
          } else if (i != length(synapse_dim) - 1 & i != 1){ #hidden layers not linked to input layer, depends of the last time step
            layers[[i]] <- logistic((layers[[i-1]]%*%time_synapse[[i]]) + (layers_values[[i]][dim(layers_values[[i]])[1],] %*% recurrent_synapse[[i]]))
          } else { # output layer depend only of the hidden layer of bellow
            layers[[i]] <- logistic(layers[[i-1]] %*% time_synapse[[i]])
          }
          
          # storing
          store[[i]][j,position,] = layers[[i]]
          if(i != length(synapse_dim) - 1){ # for all hidden layers, we need the previous state, looks like we duplicate the values here, it is also in the store list
            # store hidden layers so we can print it out. Needed for error calculation and weight iteration
            layers_values[[i]] = rbind(layers_values[[i]], layers[[i]])
          }
        }
        
        # did we miss?... if so, by how much?
        layer_2_error = y - layers[[length(synapse_dim) - 1]]
        layer_2_deltas = rbind(layer_2_deltas, layer_2_error * sigmoid_output_to_derivative(layers[[length(synapse_dim) - 1]]))
        overallError = overallError + sum(abs(layer_2_error))
        
      }
      
      # store errors
      error[j,epoch] <- overallError
      
      future_layer_delta  = list()
      for(i in seq(length(synapse_dim) - 2)){
        future_layer_delta[[i]] <- matrix(0,nrow=1, ncol = synapse_dim[i+1])
      }
      
      # Weight iteration,
      for (position in 0:(binary_dim-1)) {
        
        # input states
        x            = a[pos_vec_back[position+1],]
        # error at output layer
        layer_up_delta = layer_2_deltas[dim(layer_2_deltas)[1]-position,]
        
        for(i in (length(synapse_dim) - 1):1){
          if(i != 1){ # need update for time and recurrent synapse
            layer_current      = layers_values[[i-1]][dim(layers_values[[i-1]])[1]-position,]
            prev_layer_current = layers_values[[i-1]][dim(layers_values[[i-1]])[1]-(position+1),]
            # error at hidden layers
            layer_current_delta = (future_layer_delta[[i-1]] %*% t(recurrent_synapse[[i-1]]) + layer_up_delta %*% t(time_synapse[[i]])) *
              sigmoid_output_to_derivative(layer_current)
            time_synapse_update[[i]] = time_synapse_update[[i]] + matrix(layer_current) %*% layer_up_delta
            recurrent_synapse_update[[i-1]] = recurrent_synapse_update[[i-1]] + matrix(prev_layer_current) %*% layer_current_delta
            layer_up_delta = layer_current_delta
            future_layer_delta[[i-1]] = layer_current_delta
          }else{ # need only update for time synapse
            time_synapse_update[[i]] = time_synapse_update[[i]] + c(x) %*% layer_up_delta
          }
        }
      }
      
      
      
      # Calculate the real update including learning rate
      time_synapse_update = lapply(time_synapse_update,function(x){x* learningrate})
      recurrent_synapse_update = lapply(recurrent_synapse_update,function(x){x* learningrate})
      
      # Applying the update
      for(i in seq(length(synapse_dim) - 1)){
        time_synapse[[i]] <- time_synapse[[i]] + time_synapse_update[[i]]
      }
      for(i in seq(length(hidden_dim))){
        recurrent_synapse[[i]] <- recurrent_synapse[[i]] + recurrent_synapse_update[[i]]
      }
      
      # Update the learning rate
      learningrate <- learningrate * learningrate_decay
      
      # Initializing the update with the momentum
      time_synapse_update = lapply(time_synapse_update,function(x){x* momentum})
      recurrent_synapse_update = lapply(recurrent_synapse_update,function(x){x* momentum})
    }
    # update best guess if error is minimal
    if(colMeans(error)[epoch] <= min(colMeans(error)[1:epoch])){
      store_best <- store
    }
    message(paste0("Epoch error: ",colMeans(error)[epoch]))
  }
  
  # create utput object
  output=list(time_synapse      = time_synapse,
              recurrent_synapse = recurrent_synapse,
              error             = error,
              store             = store,
              store_best        = store_best,
              start_from_end    = start_from_end)
  
  attr(output, 'error') <- colMeans(error)
  
  
  # return output
  return(output)
  
}
