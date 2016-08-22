#' @name init_r
#' @export
#' @title init_r
#' @description Initialize the weight parameters
#' @param model the output model object
#' @return the updated model

init_r = function(model){
  if(model$network_type == "rnn"){
    init_rnn(model)
  } else if (model$network_type == "lstm"){
    init_lstm(model)
  }else{
    stop("network_type_unknown for the init")
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

#' @name init_lstm
#' @export
#' @title init_lstm
#' @description Initialize the weight parameter for a lstm
#' @param model the output model object
#' @return the updated model

init_lstm = function(model){
  if(length(model$hidden_dim) != 1){stop("only one layer LSTM supported yet")}
  
  # model$time_synapse            = list()
  # model$recurrent_synapse       = list()
  # model$bias_synapse            = list()
  # 
  # for(i in seq(length(model$hidden_dim))){
  #   model$recurrent_synapse[[i]] <- array(runif(n = model$hidden_dim[i]*model$hidden_dim[i]*4, min=-1, max=1), dim=c(model$hidden_dim[i],model$hidden_dim[i],4))
  #   model$time_synapse[[i]] <- array(runif(n = model$synapse_dim[i]*model$synapse_dim[i+1]*4, min=-1, max=1), dim=c(model$synapse_dim[i],model$synapse_dim[i+1],4))
  #   model$bias_synapse[[i]] <- matrix(runif(n = model$hidden_dim[i] * 4, min=-1, max=1), nrow=model$hidden_dim[i])
  # }
  # 
  # model$time_synapse_ouput = matrix(runif(n = model$hidden_dim[length(model$hidden_dim)]*model$output_dim, min=-1, max=1), nrow=model$hidden_dim[length(model$hidden_dim)])
  # model$bias_synapse_ouput = runif(n = model$output_dim, min=-1, max=1)
  # 
  # model$time_synapse_update            = lapply(model$time_synapse,function(x){x*0})
  # model$recurrent_synapse_update       = lapply(model$recurrent_synapse,function(x){x*0})
  # model$bias_synapse_update            = lapply(model$bias_synapse,function(x){x*0})
  # model$time_synapse_ouput_update      = model$time_synapse_ouput*0
  # model$bias_synapse_ouput_update      = model$bias_synapse_ouput*0
  
  # initialise neural network weights
  model$synapse_0_i = matrix(runif(n = model$input_dim *model$hidden_dim, min=-1, max=1), nrow=model$input_dim)
  model$synapse_0_f = matrix(runif(n = model$input_dim *model$hidden_dim, min=-1, max=1), nrow=model$input_dim)
  model$synapse_0_o = matrix(runif(n = model$input_dim *model$hidden_dim, min=-1, max=1), nrow=model$input_dim)
  model$synapse_0_c = matrix(runif(n = model$input_dim *model$hidden_dim, min=-1, max=1), nrow=model$input_dim)
  model$synapse_1   = matrix(runif(n = model$hidden_dim*model$output_dim, min=-1, max=1), nrow=model$hidden_dim)
  model$synapse_h_i = matrix(runif(n = model$hidden_dim*model$hidden_dim, min=-1, max=1), nrow=model$hidden_dim)
  model$synapse_h_f = matrix(runif(n = model$hidden_dim*model$hidden_dim, min=-1, max=1), nrow=model$hidden_dim)
  model$synapse_h_o = matrix(runif(n = model$hidden_dim*model$hidden_dim, min=-1, max=1), nrow=model$hidden_dim)
  model$synapse_h_c = matrix(runif(n = model$hidden_dim*model$hidden_dim, min=-1, max=1), nrow=model$hidden_dim)
  model$synapse_b_1 = runif(n = model$output_dim, min=-1, max=1)
  model$synapse_b_i = runif(n = model$hidden_dim, min=-1, max=1)
  model$synapse_b_f = runif(n = model$hidden_dim, min=-1, max=1)
  model$synapse_b_o = runif(n = model$hidden_dim, min=-1, max=1)
  model$synapse_b_c = runif(n = model$hidden_dim, min=-1, max=1)
  
  # initialise synapse updates
  model$synapse_0_i_update = matrix(0, nrow = model$input_dim, ncol = model$hidden_dim)
  model$synapse_0_f_update = matrix(0, nrow = model$input_dim, ncol = model$hidden_dim)
  model$synapse_0_o_update = matrix(0, nrow = model$input_dim, ncol = model$hidden_dim)
  model$synapse_0_c_update = matrix(0, nrow = model$input_dim, ncol = model$hidden_dim)
  model$synapse_1_update   = matrix(0, nrow = model$hidden_dim, ncol = model$output_dim)
  model$synapse_h_i_update = matrix(0, nrow = model$hidden_dim, ncol = model$hidden_dim)
  model$synapse_h_f_update = matrix(0, nrow = model$hidden_dim, ncol = model$hidden_dim)
  model$synapse_h_o_update = matrix(0, nrow = model$hidden_dim, ncol = model$hidden_dim)
  model$synapse_h_c_update = matrix(0, nrow = model$hidden_dim, ncol = model$hidden_dim)
  model$synapse_b_1_update = rep(0, model$output_dim)
  model$synapse_b_i_update = rep(0, model$hidden_dim)
  model$synapse_b_f_update = rep(0, model$hidden_dim)
  model$synapse_b_o_update = rep(0, model$hidden_dim)
  model$synapse_b_c_update = rep(0, model$hidden_dim)
  
  return(model)
}
