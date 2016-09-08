#' @name clean_r
#' @title init_r
#' @description Initialize the weight parameters
#' @param model the output model object
#' @return the updated model

clean_r = function(model){
  if(model$network_type == "rnn"){
    clean_rnn(model)
  } else if (model$network_type == "lstm" | model$network_type == "gru" ){
    clean_lstm(model)
  }else{
    stop("network_type_unknown for the cleaning")
  }
}

#' @name clean_rnn
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

#' @name clean_lstm
#' @title clean_lstm
#' @description clean the model for lighter output
#' @param model the output model object
#' @return the updated model

clean_lstm = function(model){
  
  return(model)
}


