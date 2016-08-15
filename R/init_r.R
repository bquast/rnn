#' @name init_r
#' @export
#' @title init_r
#' @description Initialize the weight parameters
#' @param model the output model object
#' @return the updated model

init_r = function(model){
  if(model$network_type == "rnn"){
    init_rnn(model)
  }else{
    stop("only rnn supported for the moment")
  }
}

#' @name init_rnn
#' @export
#' @title init_rnn
#' @description Initialize the weight parameter for a rnn
#' @param model the output model object
#' @return the updated model

init_rnn = function(model){
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
