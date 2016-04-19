#' @name predictr
#' @export
#' @importFrom stats runif
#' @title Recurrent Neural Network
#' @description Trains a Recurrent Neural Network.
#' @param model output of the trainr function
#' @param X1 vector of input values
#' @param X2 vector of input values
#' @param binary_dim dimension of binary representation
#' @param alpha size of alpha
#' @param input_dim dimension of input layer, i.e. how many numbers to sum
#' @param hidden_dim dimension of hidden layer
#' @param output_dim dimension of output layer
#' @param print should train progress be printed
#' @return vector of predicted values
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
#' # train the model
#' m1 <- trainr(Y,
#'              X1,
#'              X2,
#'              binary_dim =  8,
#'              alpha      =  0.1,
#'              input_dim  =  2,
#'              hidden_dim = 10,
#'              output_dim =  1   )
#'              
#' # create test inputs
#' A1 = int2bin( sample(0:127, 7000, replace=TRUE) )
#' A2 = int2bin( sample(0:127, 7000, replace=TRUE) )
#'     
#' # predict
#' B  <- predictr(m1,
#'                A1,
#'                A2,
#'                binary_dim =  8,
#'                alpha      =  0.1,
#'                input_dim  =  2,
#'                hidden_dim = 10,
#'                output_dim =  1   )
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
#' hist( B-(A1+A2) )
#' 


predictr <- function(model, X1, X2, binary_dim, alpha, input_dim, hidden_dim, output_dim, print = c('none', 'minimal', 'full')) {
  
  # check what largest possible number is
  largest_number = 2^binary_dim
  
  # create output vector
  Y <- matrix(nrow = dim(X1)[1], ncol = binary_dim)

  # load neural network weights
  synapse_0 = model$synapse_0
  synapse_1 = model$synapse_1
  synapse_h = model$synapse_h
  
  # training logic
  for (j in 1:dim(X1)[1]) {
    
    if(print != 'none' && j %% 1000 == 0) {
      print(paste('Summation number:', j))
    }
    
    # generate a simple addition problem (a + b = c)
    a = X1[j,]
    b = X2[j,]
    
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
      
      # hidden layer (input ~+ prev_hidden)
      layer_1 = sigmoid((X%*%synapse_0) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h))
      
      # output layer (new binary representation)
      layer_2 = sigmoid(layer_1 %*% synapse_1)
      
      # decode estimate so we can print it out
      d[binary_dim - position] = round(layer_2)
      
      # store hidden layer so we can print it out
      layer_1_values = rbind(layer_1_values, layer_1)
      
      if(print == 'full' && j %% 1000 == 0) {
        print(paste('x1:', a[binary_dim - position]))
        print(paste('x2:', b[binary_dim - position], '+'))
        print('-------')
        print(paste('y^:', d[binary_dim - position]))
        print('=======')
      }
    }
    
    # output to decimal
    out = b2i(as.vector(d))

    # print out progress
    if(print != 'none' && j %% 1000 == 0) {
      print(paste('Error:', overallError))
      print(paste('X1[', j, ']:', paste(a, collapse = ' '), ' ', '(', b2i(a), ')'))
      print(paste('X2[', j, ']:', paste(b, collapse = ' '), '+', '(', b2i(b), ')'))
      print('-----------------------------')
      print(paste('predict Y^:',   paste(d, collapse = ' '), ' ', '(', out, ')'))
      print('=============================')
    }
    
    # store value
    Y[j,] <- d
  }
  
  # return output vector
  return( Y )
}
