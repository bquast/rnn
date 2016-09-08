#' @name init_r
#' @title init_r
#' @description Initialize the weight parameters
#' @param model the output model object
#' @return the updated model

init_r = function(model){
  if(model$network_type == "rnn"){
    init_rnn(model)
  } else if (model$network_type == "lstm"){
    init_lstm(model)
  }else if (model$network_type == "gru"){
    init_gru(model)
  }
}

#' @name init_rnn
#' @title init_rnn
#' @description Initialize the weight parameter for a rnn
#' @param model the output model object
#' @return the updated model

init_rnn = function(model){
  
  # Storing layers states, filled with 0 for the moment
  model$store <- list()
  for(i in seq(length(model$synapse_dim) - 1)){
    model$store[[i]] <- array(0,dim = c(dim(model$last_layer_error)[1:2],model$synapse_dim[i+1]))
  }
  
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
  
  return(model)
}

#' @name init_lstm
#' @title init_lstm
#' @description Initialize the weight parameter for a lstm
#' @param model the output model object
#' @return the updated model

init_lstm = function(model){
  # if(length(model$hidden_dim) != 1){stop("only one layer LSTM supported yet")}
  
  # Storing layers states, filled with 0 for the moment
  model$store <- list()
  model$time_synapse            = list()
  model$recurrent_synapse       = list()
  model$bias_synapse            = list()
  for(i in seq(length(model$hidden_dim))){
    # hidden output / cells / forget / input / gate / output
    model$store[[i]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$hidden_dim[1],6)) # 4D arrays !!! with dim()[4] = 6
    model$time_synapse[[i]] = array(runif(n = model$synapse_dim[i] * model$synapse_dim[i+1] * 4, min=-1, max=1),dim = c(model$synapse_dim[i], model$synapse_dim[i+1], 4))# 3D arrays with dim()[3] = 4
    model$recurrent_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * model$synapse_dim[i+1] * 4, min=-1, max=1),dim = c(model$synapse_dim[i+1], model$synapse_dim[i+1], 4))# 3D arrays with dim()[3] = 4
    model$bias_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * 4, min=-1, max=1),dim = c(model$synapse_dim[i+1], 4))#2D arrays with dim()[2] = 4
  }
  
  model$store[[length(model$store) + 1]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$output_dim)) # final output layer
  model$time_synapse[[length(model$time_synapse) + 1]] = array(runif(n = model$hidden_dim[length(model$hidden_dim)] * model$output_dim, min=-1, max=1),dim = c(model$hidden_dim[length(model$hidden_dim)], model$output_dim)) # 4D arrays !!!
  model$bias_synapse[[length(model$bias_synapse) + 1]] = runif(model$output_dim,min=-0.1,max=0.1)
  
  # add the update to the model list
  model$time_synapse_update = lapply(model$time_synapse,function(x){x*0})
  model$bias_synapse_update = lapply(model$bias_synapse,function(x){x*0})
  model$recurrent_synapse_update = lapply(model$recurrent_synapse,function(x){x*0})
  
  return(model)
}

#' @name init_gru
#' @title init_gru
#' @description Initialize the weight parameter for a gru
#' @param model the output model object
#' @return the updated model

init_gru = function(model){
  
  # Storing layers states, filled with 0 for the moment
  model$store <- list()
  model$time_synapse            = list()
  model$recurrent_synapse       = list()
  model$bias_synapse            = list()
  for(i in seq(length(model$hidden_dim))){
    # hidden output / z / r / h
    model$store[[i]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$hidden_dim[1],4)) # 4D arrays !!! with dim()[4] = 4
    model$time_synapse[[i]] = array(runif(n = model$synapse_dim[i] * model$synapse_dim[i+1] * 3, min=-1, max=1),dim = c(model$synapse_dim[i], model$synapse_dim[i+1], 3))# 3D arrays with dim()[3] = 3
    model$recurrent_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * model$synapse_dim[i+1] * 3, min=-1, max=1),dim = c(model$synapse_dim[i+1], model$synapse_dim[i+1], 3))# 3D arrays with dim()[3] = 3
    model$bias_synapse[[i]] = array(runif(n = model$synapse_dim[i+1] * 3, min=-1, max=1),dim = c(model$synapse_dim[i+1], 3))#2D arrays with dim()[2] = 3
  }
  
  model$store[[length(model$store) + 1]] = array(0,dim = c(dim(model$last_layer_error)[1:2],model$output_dim)) # final output layer
  model$time_synapse[[length(model$time_synapse) + 1]] = array(runif(n = model$hidden_dim[length(model$hidden_dim)] * model$output_dim, min=-1, max=1),dim = c(model$hidden_dim[length(model$hidden_dim)], model$output_dim)) # 4D arrays !!!
  model$bias_synapse[[length(model$bias_synapse) + 1]] = runif(model$output_dim,min=-0.1,max=0.1)
  
  # add the update to the model list
  model$time_synapse_update = lapply(model$time_synapse,function(x){x*0})
  model$bias_synapse_update = lapply(model$bias_synapse,function(x){x*0})
  model$recurrent_synapse_update = lapply(model$recurrent_synapse,function(x){x*0})
  
  return(model)
}
