#' @name rnn
#' @export
#' @importFrom stats runif
#' @title Recurrent Neural Network
#' @description Trains a Recurrent Neural Network.
#' @param binary_dim dimension of binary representation
#' @param alpha size of alpha
#' @param input_dim dimension of input layer, i.e. how many numbers to sum
#' @param hidden_dim dimension of hidden layer
#' @param output_dim dimension of output layer
#' @param iterations number of training iterations, default is 10,000
#' @examples 
#' # using the default of 10,000 iterations
#' rnn(binary_dim =  8,
#'     alpha      =  0.1,
#'     input_dim  =  2,
#'     hidden_dim = 10,
#'     output_dim =  1  )


rnn <- function(binary_dim, alpha, input_dim, hidden_dim, output_dim, iterations=5000) {
  
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
  for (j in 1:iterations) {
    
    # generate a simple addition problem (a + b = c)
    a_int = sample(1:(largest_number/2), 1) # int version
    a = int2binary(a_int, binary_dim) # binary encoding
    
    b_int = sample(1:(largest_number/2), 1) # int version
    b = int2binary(b_int, binary_dim)
    
    # true answer
    c_int = a_int + b_int
    c = int2binary(c_int, binary_dim)
    
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
    
    # print out progress
    if(j %% 500 ==0) {
      print(paste("Error:", overallError))
      print(paste("Pred:", paste(d, collapse = " ")))
      print(paste("True:", paste(c, collapse = " ")))
      out = 0
      for (x in 1:length(d)) {
        out[x] = rev(d)[x]*2^(x-1) }
      print(paste(a_int, "+", b_int, "=", sum(out)))
      print("----------------")                     
    }             
  }
}
