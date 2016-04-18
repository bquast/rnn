#' @name trainr
#' @export
#' @importFrom stats runif
#' @title Recurrent Neural Network
#' @description Trains a Recurrent Neural Network.
#' @param Y vector of output values
#' @param X1 vector of input values
#' @param X2 vector of input values
#' @param binary_dim dimension of binary representation
#' @param alpha size of alpha
#' @param input_dim dimension of input layer, i.e. how many numbers to sum
#' @param hidden_dim dimension of hidden layer
#' @param output_dim dimension of output layer
#' @param print should train progress be printed
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
#' # train the model
#' trainr(Y,
#'        X1,
#'        X2,
#'        binary_dim =  8,
#'        alpha      =  0.1,
#'        input_dim  =  2,
#'        hidden_dim = 10,
#'        output_dim =  1,
#'        print = 'full'    )
#'     


trainr <- function(Y, X1, X2, binary_dim, alpha, input_dim, hidden_dim, output_dim, print = c('none', 'minimal', 'full')) {
  
  # check what largest possible number is
  largest_number = 2^binary_dim
  
  # initialize neural network weights
  synapse_0 = matrix(stats::runif(n = input_dim*hidden_dim, min=-1, max=1), nrow=input_dim)
  synapse_1 = matrix(stats::runif(n = hidden_dim*output_dim, min=-1, max=1), nrow=hidden_dim)
  synapse_h = matrix(stats::runif(n = hidden_dim*hidden_dim, min=-1, max=1), nrow=hidden_dim)
  
  synapse_0_update = matrix(0, nrow = input_dim, ncol = hidden_dim)
  synapse_1_update = matrix(0, nrow = hidden_dim, ncol = output_dim)
  synapse_h_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)
  
  # training logic
  for (j in 1:dim(Y)[1]) {
    
    if(print != 'none' && j %% 1000 == 0) {
      print(paste('Summation number:', j))
    }
    
    # generate a simple addition problem (a + b = c)
    a = X1[j,]
    b = X2[j,]
    
    # true answer
    c = Y[j,]
    
    # where we'll store our best guesss (binary encoded)
    d = matrix(0, nrow = 1, ncol = binary_dim)
    
    overallError = 0
    
    layer_2_deltas = matrix(0)
    layer_1_values = matrix(0, nrow=1, ncol = hidden_dim)
    # layer_1_values = rbind(layer_1_values, matrix(0, nrow=1, ncol=hidden_dim))
    
    # moving along the positions in the binary encoding
    for (position in 0:(binary_dim-1)) {
      
      # generate input and output
      X = cbind(a[binary_dim - position],b[binary_dim - position])
      y = c[binary_dim - position]
      
      # hidden layer (input ~+ prev_hidden)
      layer_1 = sigmoid((X%*%synapse_0) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h))
      
      # output layer (new binary representation)
      layer_2 = sigmoid(layer_1 %*% synapse_1)
      
      # did we miss?... if so, by how much?
      layer_2_error = y - layer_2
      layer_2_deltas = rbind(layer_2_deltas, layer_2_error * sigmoid_output_to_derivative(layer_2))
      overallError = overallError + abs(layer_2_error)
      
      # decode estimate so we can print it out
      d[binary_dim - position] = round(layer_2)
      
      # store hidden layer so we can print it out
      layer_1_values = rbind(layer_1_values, layer_1)
      
      if(print == 'full' && j %% 1000 == 0) {
        print(paste('x1:', a[binary_dim - position]))
        print(paste('x2:', b[binary_dim - position], '+'))
        print('-------')
        print(paste('y: ', c[binary_dim - position]))
        print(paste('y^:', d[binary_dim - position]))
        print('=======')
      }
    }
    
    future_layer_1_delta = matrix(0, nrow = 1, ncol = hidden_dim)
    
    for (position in 0:(binary_dim-1)) {
      
      X = cbind(a[position+1], b[position+1])
      layer_1 = layer_1_values[dim(layer_1_values)[1]-position,]
      prev_layer_1 = layer_1_values[dim(layer_1_values)[1]-(position+1),]
      
      # error at output layer
      layer_2_delta = layer_2_deltas[dim(layer_2_deltas)[1]-position,]
      # error at hidden layer
      layer_1_delta = (future_layer_1_delta %*% t(synapse_h) + layer_2_delta %*% t(synapse_1)) *
        sigmoid_output_to_derivative(layer_1)
      
      # let's update all our weights so we can try again
      synapse_1_update = synapse_1_update + matrix(layer_1) %*% layer_2_delta
      synapse_h_update = synapse_h_update + matrix(prev_layer_1) %*% layer_1_delta
      synapse_0_update = synapse_0_update + t(X) %*% layer_1_delta
      
      future_layer_1_delta = layer_1_delta
    }
    
    synapse_0 = synapse_0 + ( synapse_0_update * alpha )
    synapse_1 = synapse_1 + ( synapse_1_update * alpha )
    synapse_h = synapse_h + ( synapse_h_update * alpha )
    
    synapse_0_update = synapse_0_update * 0
    synapse_1_update = synapse_1_update * 0
    synapse_h_update = synapse_h_update * 0
    
    # convert d to decimal
    out = b2i(as.vector(d))
    
    # print out progress
    if(print != 'none' && j %% 1000 == 0) {
      print(paste('Error:', overallError))
      print(paste('X1[', j, ']:', paste(a, collapse = ' '), ' ', '(', b2i(a), ')'))
      print(paste('X2[', j, ']:', paste(b, collapse = ' '), '+', '(', b2i(b), ')'))
      print('-----------------------------')
      print(paste('Y[', j, ']: ', paste(c, collapse = ' '), ' ', '(', b2i(c), ')'))
      print(paste('predict Y^:',   paste(d, collapse = ' '), ' ', '(', out, ')'))
      print('=============================')
    }             
  }
  
  # output object with synapses
  return(list(synapse_0 = synapse_0, synapse_1 = synapse_1, synapse_h = synapse_h))
  
}
