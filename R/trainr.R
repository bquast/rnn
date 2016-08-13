#' @name trainr
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid sigmoid_output_to_derivative
#' @title Recurrent Neural Network
#' @description Trains a Recurrent Neural Network.
#' @param Y array of output values, dim 1: samples (must be equal to dim 1 of X), dim 2: time (must be equal to dim 2 of X), dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param learningrate learning rate to be applied for weight iteration
#' @param numepochs number of iteration, i.e. number of time the whole dataset is presented to the network
#' @param hidden_dim dimension(s) of hidden layer(s)
#' @param network_type type of network, could be rnn or lstm, only rnn supported for the moment
#' @param sigmoid method to be passed to the sigmoid function
#' @param start_from_end should the sequence start from the end, legacy of the binary example
#' @param learningrate_decay coefficient to apply to the learning rate at each epoch, via the epoch_annealing function
#' @param momentum coefficient of the last weight iteration to keep for faster learning
#' @param use_bias should the network use bias
#' @param epoch_function vector of functions with no side effect on the model to applied at each epoch loop
#' @param epoch_model_function vector of functions with side effect on the model to applied at each epoch loop
#' @param loss_function loss function, applied in each sample loop, vocabulary to verify
#' @param ... Arguments to be passed to methods, a way to store them in the model list is needed
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

trainr <- function(Y, X, learningrate, learningrate_decay = 1, momentum = 0, hidden_dim = c(10),network_type = "rnn",
                   numepochs = 1, sigmoid = c('logistic', 'Gompertz', 'tanh'), start_from_end=FALSE, use_bias = F,
                   epoch_function = c(epoch_print),
                   epoch_model_function = c(epoch_annealing),
                   loss_function = loss_L1,...) {
  
  #  find sigmoid
  sigmoid <- match.arg(sigmoid)
  
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
  
  # reverse the time dim if start from end
  if(start_from_end){
    X <- X[,dim(X)[2]:1,,drop = F]
    Y <- Y[,dim(X)[2]:1,,drop = F]
  }
  
  # initialize the model list
  model                         = list(...) # we start by the ... argument before appending everybody else
  model$input_dim               = dim(X)[3]
  model$hidden_dim              = hidden_dim
  model$output_dim              = dim(Y)[3]
  model$synapse_dim             = c(model$input_dim,model$hidden_dim,model$output_dim)
  model$time_dim                = dim(X)[2] ## changed from binary_dim to get rid of the binary user case legacy
  model$sigmoid                 = sigmoid
  model$network_type            = network_type
  model$numepochs               = numepochs
  model$learningrate            = learningrate
  model$learningrate_decay      = learningrate_decay ## this one should be in the ... arg and be here initially but he was supply before
  model$momentum                = momentum
  model$use_bias                = use_bias
  model$start_from_end          = start_from_end
  model$epoch_function          = epoch_function
  model$epoch_model_function    = epoch_model_function
  model$loss_function           = loss_function
  model$time_synapse            = list()
  model$recurrent_synapse       = list()
  model$bias_synapse            = list()
  
  
  #initialize neural network weights, stored in several lists
  for(i in seq(length(model$synapse_dim) - 1)){
    model$time_synapse[[i]] <- matrix(runif(n = model$synapse_dim[i]*model$synapse_dim[i+1], min=-1, max=1), nrow=model$synapse_dim[i])
  }
  for(i in seq(length(model$hidden_dim))){
    model$recurrent_synapse[[i]] <- matrix(runif(n = model$hidden_dim[i]*model$hidden_dim[i], min=-1, max=1), nrow=model$hidden_dim[i])
  }
  for(i in seq(length(model$synapse_dim) - 1)){
    model$bias_synapse[[i]] <- runif(model$synapse_dim[i+1],min=-0.1,max=0.1)
  }
  
  # add the update to the model list
  model$time_synapse_update = lapply(model$time_synapse,function(x){x*0})
  model$bias_synapse_update = lapply(model$bias_synapse,function(x){x*0})
  model$recurrent_synapse_update = lapply(model$recurrent_synapse,function(x){x*0})
  
  # Storing layers states, filled with 0 for the moment
  model$store <- list()
  for(i in seq(length(model$synapse_dim) - 1)){
    model$store[[i]] <- array(0,dim = c(dim(Y)[1:2],model$synapse_dim[i+1]))
  }
  
  # Storing errors, dim 1: samples, dim 2 is epochs, we could store also the time and variable dimension
  model$error <- array(0,dim = c(dim(Y)[1],model$numepochs))
  
  # training logic
  for(epoch in seq(model$numepochs)){
    model$current_epoch = epoch
    for (j in 1:dim(Y)[1]) {
      
      # generate input and output for the sample loop
      a = array(X[j,,],dim=c(dim(X)[2],model$input_dim))
      c = array(Y[j,,],dim=c(dim(Y)[2],model$output_dim))
      
      overallError = 0
      layer_2_deltas = matrix(0,nrow=1, ncol = model$output_dim)
      # store the hidden layers values for each time step, needed in parallel of store because we need the t(-1) hidden states. otherwise, we could take the values from the store list
      layers_values  = list()
      for(i in seq(length(model$synapse_dim) - 2)){
        layers_values[[i]] <- matrix(0,nrow=1, ncol = model$synapse_dim[i+1])
      }
      
      # time index vector, needed because we predict in one direction but update the weight in an other
      pos_vec      <- 1:model$time_dim
      pos_vec_back <- model$time_dim:1
      
      # moving along the time
      for (position in pos_vec) {
        
        # generate input and output for the position loop
        x = a[position,]
        y = c[position,]
        
        layers <- list()
        for(i in seq(length(model$synapse_dim) - 1)){
          if (i == 1) { # first hidden layer, need to take x as input
            layers[[i]] <- (x%*%model$time_synapse[[i]]) + (layers_values[[i]][dim(layers_values[[i]])[1],] %*% model$recurrent_synapse[[i]])
          } else if (i != length(model$synapse_dim) - 1 & i != 1){ #hidden layers not linked to input layer, depends of the last time step
            layers[[i]] <- (layers[[i-1]]%*%model$time_synapse[[i]]) + (layers_values[[i]][dim(layers_values[[i]])[1],] %*% model$recurrent_synapse[[i]])
          } else { # output layer depend only of the hidden layer of bellow
            layers[[i]] <- layers[[i-1]] %*% model$time_synapse[[i]]
          }
          if(model$use_bias){ # apply the bias if applicable
            layers[[i]] <- layers[[i]] + model$bias_synapse[[i]]
          }
          # apply the activation function. user specify function ? he will need to supply the derivative also ??
          layers[[i]] <- sigmoid(layers[[i]], method=model$sigmoid) 
          
          # storing
          model$store[[i]][j,position,] = layers[[i]]
          if(i != length(model$synapse_dim) - 1){ # for all hidden layers, we need the previous state, looks like we duplicate the values here, it is also in the store list
            # store hidden layers so we can print it out. Needed for error calculation and weight iteration
            layers_values[[i]] = rbind(layers_values[[i]], layers[[i]])
          }
        }
        
        # did we miss?... if so, by how much?
        layer_2_error = y - layers[[length(model$synapse_dim) - 1]]
        layer_2_deltas = rbind(layer_2_deltas, layer_2_error * sigmoid_output_to_derivative(layers[[length(model$synapse_dim) - 1]])) ## here I think we need a specific derivative, specially if user specify function
        overallError = overallError + sum(abs(layer_2_error))
        
      } # end position feed forward loop
      
      # store errors
      model$error[j,model$current_epoch] <- overallError
      
      future_layer_delta  = list()
      for(i in seq(length(model$synapse_dim) - 2)){
        future_layer_delta[[i]] <- matrix(0,nrow=1, ncol = model$synapse_dim[i+1])
      }
      
      # Weight iteration,
      for (position in 0:(model$time_dim-1)) {
        
        # input states
        x            = a[pos_vec_back[position+1],]
        # error at output layer
        layer_up_delta = array(layer_2_deltas[dim(layer_2_deltas)[1]-position,],dim=c(1,model$output_dim)) # arrray dimension because of bias colMeans function on layer_up_delta
        
        for(i in (length(model$synapse_dim) - 1):1){
          if(i != 1){ # need update for time and recurrent synapse
            layer_current      = layers_values[[i-1]][dim(layers_values[[i-1]])[1]-position,]
            prev_layer_current = layers_values[[i-1]][dim(layers_values[[i-1]])[1]-(position+1),]
            # error at hidden layers
            layer_current_delta = (future_layer_delta[[i-1]] %*% t(model$recurrent_synapse[[i-1]]) + layer_up_delta %*% t(model$time_synapse[[i]])) *
              sigmoid_output_to_derivative(layer_current)
            model$time_synapse_update[[i]] = model$time_synapse_update[[i]] + matrix(layer_current) %*% layer_up_delta
            model$bias_synapse_update[[i]] = model$bias_synapse_update[[i]] + colMeans(layer_up_delta)
            model$recurrent_synapse_update[[i-1]] = model$recurrent_synapse_update[[i-1]] + matrix(prev_layer_current) %*% layer_current_delta
            layer_up_delta = layer_current_delta
            future_layer_delta[[i-1]] = layer_current_delta
          }else{ # need only update for time synapse
            model$time_synapse_update[[i]] = model$time_synapse_update[[i]] + c(x) %*% layer_up_delta
          }
        }
      } # end position back prop loop
      
      
      
      # apply the loss function, default is to apply L1 learning rate, vocabulary to verify.
      model = model$loss_function(model)
      
      # Applying the update
      for(i in seq(length(model$synapse_dim) - 1)){
        model$time_synapse[[i]] <- model$time_synapse[[i]] + model$time_synapse_update[[i]]
        model$bias_synapse[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
      }
      for(i in seq(length(model$hidden_dim))){
        model$recurrent_synapse[[i]] <- model$recurrent_synapse[[i]] + model$recurrent_synapse_update[[i]]
      }
      
      # Initializing the update with the momentum
      model$time_synapse_update = lapply(model$time_synapse_update,function(x){x* model$momentum})
      model$bias_synapse_update = lapply(model$bias_synapse_update,function(x){x* model$momentum})
      model$recurrent_synapse_update = lapply(model$recurrent_synapse_update,function(x){x* model$momentum})
    } # end sample loop
    
    
    # update best guess if error is minimal, will make more sens to store the weight...
    if(colMeans(model$error)[epoch] <= min(colMeans(model$error)[1:epoch])){
      model$store_best <- model$store
    }
    
    # # epoch_model_function
    for(i in model$epoch_model_function){
      model <- i(model)
    }
    
    # epoch_function
    for(i in model$epoch_function){
      i(model)
    }
    
  } # end epoch loop
  
  # clean model object, get rid of the update mainly, potentially other cleaning if not necessary in predictr
  model$time_synapse_update      = NULL
  model$bias_synapse_update      = NULL
  model$recurrent_synapse_update = NULL
  if(model$use_bias != T){model$bias_synapse = NULL}
  model$current_epoch = NULL
  
  attr(model, 'error') <- colMeans(model$error)
  
  # return output
  return(model)
  
}
