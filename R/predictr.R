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
  time_synapse      = model$time_synapse
  recurrent_synapse = model$recurrent_synapse
  start_from_end    = model$start_from_end
  
  # extract the network dimensions, only the binary dim
  input_dim         = dim(time_synapse[[1]])[1]
  output_dim        = dim(time_synapse[[length(time_synapse)]])[2]
  synapse_dim        = c(unlist(lapply(time_synapse,function(x){dim(x)[1]})),output_dim)
  binary_dim         = dim(X)[2]
  
  store <- list()
  for(i in seq(length(synapse_dim) - 1)){
    store[[i]] <- array(0,dim = c(dim(X)[1:2],synapse_dim[i+1]))
  }
  
  
  for (j in 1:dim(X)[1]) {
    
    # generate a simple addition problem (a + b = c)
    a = array(X[j,,],dim=c(dim(X)[2],input_dim))
    
    
    # store the hidden layers values for each time step, needed in parallel of store because we need the t(-1) hidden states. otherwise, we could take the values from the store list
    layers_values  = list()
    for(i in seq(length(synapse_dim) - 2)){
      layers_values[[i]] <- matrix(0,nrow=1, ncol = synapse_dim[i+1])
    }
    
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
      
      layers <- list()
      for(i in seq(length(synapse_dim) - 1)){
        if(i == 1){ # first hidden layer, need to take x as input
          layers[[i]] <- logistic((x%*%time_synapse[[i]]) + (layers_values[[i]][dim(layers_values[[i]])[1],] %*% recurrent_synapse[[i]]))
        }
        if(i != length(synapse_dim) - 1 & i != 1){ #hidden layers not linked to input layer, depends of the last time step
          layers[[i]] <- logistic((layers[[i-1]]%*%time_synapse[[i]]) + (layers_values[[i]][dim(layers_values[[i]])[1],] %*% recurrent_synapse[[i]]))
        }
        if(i == length(synapse_dim) - 1){ # output layer depend only of the hidden layer of bellow
          layers[[i]] <- logistic(layers[[i-1]] %*% time_synapse[[i]])
        }
        # storing
        store[[i]][j,position,] = layers[[i]]
        if(i != length(synapse_dim) - 1){ # for all hidden layers, we need the previous state, looks like we duplicate the values here, it is also in the store list
          # store hidden layers so we can print it out. Needed for error calculation and weight iteration
          layers_values[[i]] = rbind(layers_values[[i]], layers[[i]])
        }
      }
    }
  }
  # convert output to matrix if 2 dimensional
    if(dim(store[[length(store)]])[3]==1) {
      store[[length(store)]] <- matrix(store[[length(store)]],
                       nrow = dim(store[[length(store)]])[1],
                       ncol = dim(store[[length(store)]])[2])  
      }
  
  # return output vector
  if(hidden == FALSE){ # return only the last ele;ent of the list, i.e. the output
    return(store[[length(store)]])
  }else{ # return everything
    return(store)
  }
}
