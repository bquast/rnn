#' @name clean_r
#' @export
#' @title init_r
#' @description Initialize the weight parameters
#' @param model the output model object
#' @return the updated model

clean_r = function(model){
  if(model$network_type == "rnn"){
    clean_rnn(model)
  }else{
    stop("only rnn supported for the moment")
  }
}

#' @name clean_rnn
#' @export
#' @title clean_rnn
#' @description clean the model for lighter output
#' @param model the output model object
#' @return the updated model

clean_rnn = function(model){
  model$time_synapse_update      = NULL
  model$bias_synapse_update      = NULL
  model$recurrent_synapse_update = NULL
  if(model$use_bias != T){model$bias_synapse = NULL}
  model$current_epoch = NULL
  
  return(model)
}
