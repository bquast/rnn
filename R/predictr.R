#' @name predictr
#' @export
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
#' 
predictr = function(model, X, hidden = FALSE, real_output = T,...){
  
  # coerce to array if matrix
  if(length(dim(X)) == 2){
    X <- array(X,dim=c(dim(X),1))
  }
  
  if(model$network_type == "rnn"){
    predict_rnn(model, X, hidden, real_output,...)
  } else if (model$network_type == "lstm"){
    predict_lstm(model, X, hidden, real_output,...)
  }else{
    stop("network_type_unknown for the prediction")
  }
}

#' @name predict_rnn
#' @export
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
  
  for (position in 1:model$time_dim) {
    
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
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid
#' @title LSTM prediction function
#' @description predict the output of a LSTM model
#' @param model output of the trainr function
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param hidden should the function output the hidden units states
#' @param real_output option used when the function in called inside trainr, do not drop factor for 2 dimension array output
#' @param ... arguments to pass on to sigmoid function
#' @return array or matrix of predicted values

predict_lstm <- function(model, X, hidden = FALSE, real_output = T,...) {
  # store <- list(X)
  # layer_1_values = list()
  # c_t = list()
  # for(i in seq(length(model$hidden_dim))){
  #   store[[i+1]] <- array(0,dim = c(dim(X)[1:2],model$hidden_dim[i]))
  #   layer_1_values[[i]]  = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i])
  #   c_t[[i]]         = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i])
  # }
  # store[[length(store)+1]] <- array(0,dim = c(dim(X)[1:2],model$output_dim))
  # 
  # for (position in 1:model$time_dim) {
  #   
  #   for(i in seq(length(model$hidden_dim))){
  #     
  #     # generate input 
  #     x = array(store[[i]][,position,],dim=dim(store[[i]])[c(1,3)])
  #     
  #     # hidden layer (input ~+ prev_hidden)
  #     i_t     = sigmoid((x %*% model$time_synapse[[i]][,,1]) + (layer_1_values[[i]] %*% model$recurrent_synapse[[i]][,,1])) ## input gate
  #     f_t     = sigmoid((x %*% model$time_synapse[[i]][,,2]) + (layer_1_values[[i]] %*% model$recurrent_synapse[[i]][,,2])) ## forget gate
  #     o_t     = sigmoid((x %*% model$time_synapse[[i]][,,3]) + (layer_1_values[[i]] %*% model$recurrent_synapse[[i]][,,3])) ## output gate
  #     c_in_t  = tanh(   (x %*% model$time_synapse[[i]][,,4]) + (layer_1_values[[i]] %*% model$recurrent_synapse[[i]][,,4])) ## cell gate
  #     if(model$use_bias){
  #       i_t     = i_t + model$bias_synapse[[i]][,1] ## input gate
  #       f_t     = f_t + model$bias_synapse[[i]][,2] ## forget gate
  #       o_t     = o_t + model$bias_synapse[[i]][,3] ## output gate
  #       c_in_t  = c_in_t + model$bias_synapse[[i]][,4] ## cell gate
  #     }
  #     c_t[[i]]     = (f_t * c_t[[i]]) + (i_t * c_in_t)
  #     store[[i+1]][,position,] = o_t * tanh(c_t[[i]])
  #     
  #     layer_1_values[[i]] = store[[i+1]][,position,]
  #   } # end synapse loop
  #   # output layer
  #   store[[length(store)]][,position,] = store[[length(store) - 1]][,position,] %*% model$time_synapse_ouput
  #   if(model$use_bias){
  #     store[[length(store)]][,position,] = store[[length(store)]][,position,]  + model$bias_synapse_ouput
  #   }
  #   store[[length(store)]][,position,] = sigmoid(store[[length(store)]][,position,])
  # } # end time loop
  # 
  # store[[1]] <- NULL ## removed the X input
  
  store <- list()
  layer_1_values = list()
  c_t = list()
  for(i in seq(length(model$hidden_dim))){
    store[[i]] <- array(0,dim = c(dim(X)[1:2],model$hidden_dim[i]))
    layer_1_values[[i]]  = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
    c_t[[i]]         = matrix(0,nrow=dim(X)[1], ncol = model$hidden_dim[i]) # we need this object because of t-1 which do not exist in store
  }
  store[[length(store)+1]] <- array(0,dim = c(dim(X)[1:2],model$output_dim))

  for (position in 1:model$time_dim) {

      # generate input
      x = array(X[,position,],dim=dim(X)[c(1,3)])

      # hidden layer (input ~+ prev_hidden)
      i_t     = sigmoid((x %*% model$synapse_0_i) + (layer_1_values[[1]] %*% model$synapse_h_i) + model$synapse_b_i) # add bias?
      f_t     = sigmoid((x %*% model$synapse_0_f) + (layer_1_values[[1]] %*% model$synapse_h_f) + model$synapse_b_f) # add bias?
      o_t     = sigmoid((x %*% model$synapse_0_o) + (layer_1_values[[1]] %*% model$synapse_h_o) + model$synapse_b_o) # add bias?
      c_in_t  = tanh(   (x %*% model$synapse_0_c) + (layer_1_values[[1]] %*% model$synapse_h_c) + model$synapse_b_c)
      c_t[[1]]     = (f_t * c_t[[1]]) + (i_t * c_in_t)
      store[[1]][,position,] = o_t * tanh(c_t[[1]])
      
      # output layer (new binary representation)
      store[[2]][,position,] = sigmoid(store[[1]][,position,] %*% model$synapse_1 + model$synapse_b_1)
      
      # store hidden layer so we can print it out
      layer_1_values[[1]] = store[[1]][,position,]
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

