#' @name update_r
#' @export
#' @title update_r
#' @description Apply the update
#' @param model the output model object
#' @return the updated model

update_r = function(model){
  if(model$network_type == "rnn"){
    update_rnn(model)
  }else{
    stop("only rnn supported for the moment")
  }
}

#' @name update_rnn
#' @export
#' @title update_rnn
#' @description Apply the update for a rnn
#' @param model the output model object
#' @return the updated model

update_rnn = function(model){
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
  
  return(model)
}
