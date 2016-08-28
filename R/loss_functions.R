#' @name loss_L1
#' @export
#' @title L1 loss
#' @description Apply the learning rate to the weight update, vocabulary to verify !!
#' @param model the output model object
#' @return the updated model

loss_L1 = function(model){
  if(model$network_type == "rnn"){
    model$time_synapse_update = lapply(model$time_synapse_update,function(x){x* model$learningrate})
    model$bias_synapse_update = lapply(model$bias_synapse_update,function(x){x* model$learningrate})
    model$recurrent_synapse_update = lapply(model$recurrent_synapse_update,function(x){x* model$learningrate})
  } else if(model$network_type == "lstm"){
    model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,function(x){x * model$learningrate})
    model$time_synapse_update       = lapply(model$time_synapse_update,function(x){x * model$learningrate})
    model$bias_synapse_update       = lapply(model$bias_synapse_update, function(x){x * model$learningrate})
  } else if(model$network_type == "gru"){
    model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,function(x){x * model$learningrate})
    model$time_synapse_update       = lapply(model$time_synapse_update,function(x){x * model$learningrate})
    model$bias_synapse_update       = lapply(model$bias_synapse_update, function(x){x * model$learningrate})
  }
  return(model)
}

