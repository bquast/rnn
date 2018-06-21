#' @name predictr
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid
#' @title Recurrent Neural Network
#' @description predict the output of a RNN model
#' @param model output of the trainr function
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param hidden should the function output the hidden units states
#' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output and other actions. Let it to TRUE, the default, to let the function take care of the data.
#' @param ... arguments to pass on to sigmoid function
#' @return array or matrix of predicted values
#' @examples
#' \dontrun{ 
#' # create training numbers
#' X1 = sample(0:127, 10000, replace=TRUE)
#' X2 = sample(0:127, 10000, replace=TRUE)
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
#' model <- trainr(Y=Y[,dim(Y)[2]:1],
#'                 X=X[,dim(X)[2]:1,],
#'                 learningrate   =  1,
#'                 hidden_dim     = 16 )
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
#'                A[,dim(A)[2]:1,]     )
#' B = B[,dim(B)[2]:1]
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
#' }
#' 
predictr = function(model, X, hidden = FALSE, real_output = T,...){
  
  # coerce to array if matrix
  if(length(dim(X)) == 2){
    X <- array(X,dim=c(dim(X),1))
  }
  
  if(real_output && model$seq_to_seq_unsync){ ## here we modify the X in case of seq_2_seq & real_output to have the good dimensions
    time_dim_input = dim(X)[2]
    store = array(0, dim = c(dim(X)[1],model$time_dim,dim(X)[3]))
    store[,1:dim(X)[2],] = X
    X = store
    rm(store)
  }
  
  if(model$network_type == "rnn"){
    store = predict_rnn(model, X, hidden, real_output,...)
  } else if (model$network_type == "lstm"){
    store = predict_lstm(model, X, hidden, real_output,...)
  } else if (model$network_type == "gru"){
    store = predict_gru(model, X, hidden, real_output,...)
  }else{
    stop("network_type_unknown for the prediction")
  }
  
  if(real_output && model$seq_to_seq_unsync){
    if(length(dim(store)) == 2){
      store = store[,model$time_dim_input:model$time_dim,drop=F]
    }else{
      store = store[,model$time_dim_input:model$time_dim,,drop=F]
    }
  }
  
  return(store)
}

#' @name predict_rnn
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid
#' @title Recurrent Neural Network
#' @description predict the output of a RNN model
#' @param model output of the trainr function
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param hidden should the function output the hidden units states
#' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output
#' @param ... arguments to pass on to sigmoid function
#' @return array or matrix of predicted values

predict_rnn <- function(model, X, hidden = FALSE, real_output = T,...) {
  
  store <- list()
  for(i in seq(length(model$synapse_dim) - 1)){
    store[[i]] <- array(0,dim = c(dim(X)[1:2],model$synapse_dim[i+1]))
  }
  
  # store the hidden layers values for each time step, needed in parallel of store because we need the t(-1) hidden states. otherwise, we could take the values from the store list
  layers_values  = list()
  for(i in seq(length(model$synapse_dim) - 2)){
    layers_values[[i]] <- matrix(0,nrow=dim(X)[1], ncol = model$synapse_dim[i+1])
  }
  
  for (position in 1:dim(X)[2]) {
    
    # generate input 
    x = array(X[,position,],dim=dim(X)[c(1,3)])
    
    for(i in seq(length(model$synapse_dim) - 1)){
      if (i == 1) { # first hidden layer, need to take x as input
        store[[i]][,position,] <- (x %*% model$time_synapse[[i]]) + (layers_values[[i]] %*% model$recurrent_synapse[[i]])
      } else if (i != length(model$synapse_dim) - 1 & i != 1){ #hidden layers not linked to input layer, depends of the last time step
        store[[i]][,position,] <- (store[[i-1]][,position,] %*% model$time_synapse[[i]]) + (layers_values[[i]] %*% model$recurrent_synapse[[i]])
      } else { # output layer depend only of the hidden layer of bellow
        store[[i]][,position,] <- store[[i-1]][,position,] %*% model$time_synapse[[i]]
      }
      if(model$use_bias){ # apply the bias if applicable
        store[[i]][,position,] <- store[[i]][,position,] + model$bias_synapse[[i]]
      }
      # apply the activation function
      store[[i]][,position,] <- sigmoid(store[[i]][,position,], method=model$sigmoid)
      
      if(i != length(model$synapse_dim) - 1){ # for all hidden layers, we need the previous state, looks like we duplicate the values here, it is also in the store list
        # store hidden layers so we can print it out. Needed for error calculation and weight iteration
        layers_values[[i]] = store[[i]][,position,]
      }
    }
  }
  
  # convert output to matrix if 2 dimensional, real_output argument added if used inside trainr
  if(real_output){
    if(dim(store[[length(store)]])[3]==1) {
      store[[length(store)]] <- matrix(store[[length(store)]],
                                       nrow = dim(store[[length(store)]])[1],
                                       ncol = dim(store[[length(store)]])[2])
    }
  }
  
  # return output
  if(hidden == FALSE){ # return only the last element of the list, i.e. the output
    return(store[[length(store)]])
  }else{ # return everything
    return(store)
  }
}

#' @name predict_lstm
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid
#' @title gru prediction function
#' @description predict the output of a lstm model
#' @param model output of the trainr function
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param hidden should the function output the hidden units states
#' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output
#' @param ... arguments to pass on to sigmoid function
#' @return array or matrix of predicted values

predict_lstm <- function(model, X, hidden = FALSE, real_output = T,...) {
  
  store <- list()
  prev_layer_values = list()
  c_t = list()
  for(i in seq(length(model$hidden_dim))){
    store[[i]] = array(0,dim = c(dim(X)[1:2],model$hidden_dim[i],6)) # 4d arrays !!!, hidden, cell, f, i, g, o
    prev_layer_values[[i]]  = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
    c_t[[i]]         = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
  }
  store[[length(store)+1]] <- array(0,dim = c(dim(X)[1:2],model$output_dim))
  
  for (position in 1:dim(X)[2]) {
    
    # generate input
    x = array(X[,position,],dim=dim(X)[c(1,3)])
    
    for(i in seq(length(model$hidden_dim))){
      # hidden layer (input ~+ prev_hidden)
      f_t     = (x %*% array(model$time_synapse[[i]][,,1],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,1],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
      i_t     = (x %*% array(model$time_synapse[[i]][,,2],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,2],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
      c_in_t  = (x %*% array(model$time_synapse[[i]][,,3],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,3],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
      o_t     = (x %*% array(model$time_synapse[[i]][,,4],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (prev_layer_values[[i]] %*% array(model$recurrent_synapse[[i]][,,4],dim=c(dim(model$recurrent_synapse[[i]])[1:2])))
      if(model$use_bias){
        f_t = f_t + model$bias_synapse[[i]][,1]
        i_t = i_t + model$bias_synapse[[i]][,2]
        c_in_t = c_in_t + model$bias_synapse[[i]][,3]
        o_t = o_t + model$bias_synapse[[i]][,4]
      }
      f_t = sigmoid(f_t)
      i_t = sigmoid(i_t)
      c_in_t = tanh(c_in_t)
      o_t = sigmoid(o_t)
      
      c_t[[i]]     = f_t * c_t[[i]] + (i_t * c_in_t)
      store[[i]][,position,,1] = o_t * tanh(c_t[[i]])
      store[[i]][,position,,2] = c_t[[i]]
      store[[i]][,position,,3] = f_t
      store[[i]][,position,,4] = i_t
      store[[i]][,position,,5] = c_in_t
      store[[i]][,position,,6] = o_t
      
      # replace the x in case of multi layer
      prev_layer_values[[i]] = x = o_t * tanh(c_t[[i]])# the top of this layer at this position is the past of the top layer at the next position
    }
    
    
    # output layer (new binary representation)
    store[[length(store)]][,position,] = store[[length(store) - 1]][,position,,1] %*% model$time_synapse[[length(model$time_synapse)]]
    if(model$use_bias){
      store[[length(store)]][,position,] = store[[length(store)]][,position,] + model$bias_synapse[[length(model$bias_synapse)]]
    }
    store[[length(store)]][,position,] = sigmoid(store[[length(store)]][,position,])
  } # end time loop
  
  # convert output to matrix if 2 dimensional, real_output argument added if used inside trainr
  if(real_output){
    if(dim(store[[length(store)]])[3]==1) {
      store[[length(store)]] <- matrix(store[[length(store)]],
                                       nrow = dim(store[[length(store)]])[1],
                                       ncol = dim(store[[length(store)]])[2])
    }
  }
  
  # return output
  if(hidden == FALSE){ # return only the last element of the list, i.e. the output
    return(store[[length(store)]])
  }else{ # return everything
    return(store)
  }
}



#' @name predict_gru
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid
#' @title gru prediction function
#' @description predict the output of a gru model
#' @param model output of the trainr function
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param hidden should the function output the hidden units states
#' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output
#' @param ... arguments to pass on to sigmoid function
#' @return array or matrix of predicted values

predict_gru <- function(model, X, hidden = FALSE, real_output = T,...) {
  
  store <- list()
  h_t = list()
  for(i in seq(length(model$hidden_dim))){
    store[[i]] = array(0,dim = c(dim(X)[1:2],model$hidden_dim[i],4)) # 4d arrays !!!, hidden, z, r, h
    h_t[[i]]         = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
  }
  store[[length(store)+1]] <- array(0,dim = c(dim(X)[1:2],model$output_dim))
  
  for (position in 1:dim(X)[2]) {
    
    # generate input
    x = array(X[,position,],dim=dim(X)[c(1,3)])
    
    for(i in seq(length(model$hidden_dim))){
      # hidden layer (input ~+ prev_hidden)
      z_t     = (x %*% array(model$time_synapse[[i]][,,1],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (h_t[[i]]  %*% array(model$recurrent_synapse[[i]][,,1],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) 
      r_t     = (x %*% array(model$time_synapse[[i]][,,2],dim=c(dim(model$time_synapse[[i]])[1:2]))) + (h_t[[i]]  %*% array(model$recurrent_synapse[[i]][,,2],dim=c(dim(model$recurrent_synapse[[i]])[1:2])))
      if(model$use_bias){
        z_t = z_t + model$bias_synapse[[i]][,1]
        r_t = r_t + model$bias_synapse[[i]][,2]
      }
      z_t = sigmoid(z_t)
      r_t = sigmoid(r_t)
      
      h_in_t  = (x %*% array(model$time_synapse[[i]][,,3],dim=c(dim(model$time_synapse[[i]])[1:2]))) + ((h_t[[i]]  * r_t) %*% array(model$recurrent_synapse[[i]][,,3],dim=c(dim(model$recurrent_synapse[[i]])[1:2])))
      if(model$use_bias){
        h_in_t = h_in_t + model$bias_synapse[[i]][,3]
      }
      h_in_t = tanh(h_in_t)
      
      h_t[[i]]     = (1 - z_t) * h_t[[i]] + (z_t * h_in_t)
      store[[i]][,position,,1] = h_t[[i]]
      store[[i]][,position,,2] = z_t
      store[[i]][,position,,3] = r_t
      store[[i]][,position,,4] = h_in_t
      
      # replace the x in case of multi layer
      x = h_t[[i]]  # the top of this layer at this position is the past of the top layer at the next position
    }
    
    
    # output layer (new binary representation)
    store[[length(store)]][,position,] = store[[length(store) - 1]][,position,,1] %*% model$time_synapse[[length(model$time_synapse)]]
    if(model$use_bias){
      store[[length(store)]][,position,] = store[[length(store)]][,position,] + model$bias_synapse[[length(model$bias_synapse)]]
    }
    store[[length(store)]][,position,] = sigmoid(store[[length(store)]][,position,])
  } # end time loop
  
  # convert output to matrix if 2 dimensional, real_output argument added if used inside trainr
  if(real_output){
    if(dim(store[[length(store)]])[3]==1) {
      store[[length(store)]] <- matrix(store[[length(store)]],
                                       nrow = dim(store[[length(store)]])[1],
                                       ncol = dim(store[[length(store)]])[2])
    }
  }
  
  # return output
  if(hidden == FALSE){ # return only the last element of the list, i.e. the output
    return(store[[length(store)]])
  }else{ # return everything
    return(store)
  }
}

