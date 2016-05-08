#' @name predictr
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid
#' @title Recurrent Neural Network
#' @description predict the output of a RNN model
#' @param model output of the trainr function
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param hidden should the function output the hidden units states
#' @param ... arguments to pass on to sigmoid function
#' @return array or matrix of predicted values
#' @examples 
#' # create training numbers
#' X1 = sample(0:127, 7000, replace=TRUE)
#' X2 = sample(0:127, 7000, replace=TRUE)
#' 
#' # create training response numbers
#' Y <- X1 + X2
#' 
#' # convert to binary
#' X1 <- int2bin(X1)
#' X2 <- int2bin(X2)
#' Y  <- int2bin(Y)
#' 
#' # Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
#' X <- array( c(X1,X2), dim=c(dim(X1),2) )
#' 
#' # train the model
#' model <- trainr(Y=Y,
#'                 X=X,
#'                 learningrate   =  0.1,
#'                 hidden_dim     = 10,
#'                 start_from_end = TRUE )
#'              
#' # create test inputs
#' A1 = int2bin( sample(0:127, 7000, replace=TRUE) )
#' A2 = int2bin( sample(0:127, 7000, replace=TRUE) )
#' 
#' # create 3d array: dim 1: samples; dim 2: time; dim 3: variables
#' A <- array( c(A1,A2), dim=c(dim(A1),2) )
#'     
#' # predict
#' B  <- predictr(model,
#'                A     )
#'  
#' # convert back to integers
#' A1 <- bin2int(A1)
#' A2 <- bin2int(A2)
#' B  <- bin2int(B)
#'  
#' # inspect the differences              
#' table( B-(A1+A2) )
#' 
#' # plot the difference
#' hist(  B-(A1+A2) )
#' 

predictr <- function(model, X, hidden = FALSE, ...) {
  
  # coerce to array if matrix
  if(length(dim(X)) == 2){
    X <- array(X,dim=c(dim(X),1))
  }
  
  # load neural network weights
  synapse_0      = model$synapse_0
  synapse_1      = model$synapse_1
  synapse_h      = model$synapse_h
  start_from_end = model$start_from_end
  
  # extract the network dimensions, only the binary dim
  input_dim = dim(synapse_0)[1]
  output_dim = dim(synapse_1)[2]
  hidden_dim = dim(synapse_0)[2]
  binary_dim = dim(X)[2]
  
  # Storing layers states
  store_output <- array(0,dim = c(dim(X)[1:2],output_dim))
  store_hidden <- array(0,dim = c(dim(X)[1:2],hidden_dim))
  
  for (j in 1:dim(X)[1]) {
    
    # generate a simple addition problem (a + b = c)
    a = array(X[j,,],dim=c(dim(X)[2],input_dim))
    
    
    layer_1_values = matrix(0, nrow=1, ncol = hidden_dim)
    
    # time index vector, needed because we predict in one direction but update the weight in an other
    if(start_from_end == T){
      pos_vec <- binary_dim:1
      pos_vec_back <- 1:binary_dim
    }else{
      pos_vec <- 1:binary_dim
      pos_vec_back <- binary_dim:1
    }
    
    # moving along the time
    for (position in pos_vec) {
      
      # generate input and output
      x = a[position,]
      
      # hidden layer (input ~+ prev_hidden)
      layer_1 = sigmoid::sigmoid((x%*%synapse_0) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h), ...)
      
      # output layer (new binary representation)
      layer_2 = sigmoid::sigmoid(layer_1 %*% synapse_1, ...)
      
      # storing
      store_output[j,position,] = layer_2
      store_hidden[j,position,] = layer_1
      
      # store hidden layer so we can print it out. Needed for error calculation and weight iteration
      layer_1_values = rbind(layer_1_values, layer_1)
      
    }
  }

  
  # return output vector
  if(hidden == FALSE){
    # convert to matrix if 2 dimensional
    if(dim(store_output)[3]==1) {
      store_output <- matrix(store_output,
                       nrow = dim(store_output)[1],
                       ncol = dim(store_output)[2])  }
    # return output
    return(store_output)
  }else{
    return(store_hidden)
  }
}
