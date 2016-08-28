#' @name update_r
#' @export
#' @title update_r
#' @description Apply the update
#' @param model the output model object
#' @return the updated model

update_r = function(model){
  if(model$network_type == "rnn"){
    update_rnn(model)
  } else if (model$network_type == "lstm" | model$network_type == "gru" ){ # gru update is the same as lstm update
    update_lstm(model)
  }else{
    stop("network_type_unknown for the update")
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

#' @name update_lstm
#' @export
#' @title update_lstm
#' @description Apply the update for a lstm
#' @param model the output model object
#' @return the updated model

update_lstm = function(model){
  
  if(!is.null(model$clipping)){ # should we clippe the update or the weight, the update will make more sens as the weight lead to killed units
    clipping = function(x){
      x[is.nan(x)] = runif(sum(is.nan(x)),-1,1)
      x[x > model$clipping] = model$clipping
      x[x < -model$clipping] = - model$clipping
      return(x)
    }
    model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,clipping)
    model$time_synapse_update       = lapply(model$time_synapse_update,clipping)
    model$bias_synapse_update       = lapply(model$bias_synapse_update, clipping)
  }
  
  for(i in seq(length(model$hidden_dim))){
    model$recurrent_synapse[[i]] <- model$recurrent_synapse[[i]] + model$recurrent_synapse_update[[i]]
    model$time_synapse[[i]] <- model$time_synapse[[i]] + model$time_synapse_update[[i]]
    model$bias_synapse[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
  }

  model$time_synapse[[length(model$hidden_dim)+1]] <- model$time_synapse[[length(model$hidden_dim)+1]] + model$time_synapse_update[[length(model$hidden_dim)+1]]
  model$bias_synapse[[length(model$hidden_dim)+1]] <- model$bias_synapse[[length(model$hidden_dim)+1]] + model$bias_synapse_update[[length(model$hidden_dim)+1]]

  # Initializing the update with the momentum
  model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,function(x){x * model$momentum})
  model$time_synapse_update       = lapply(model$time_synapse_update,function(x){x * model$momentum})
  model$bias_synapse_update       = lapply(model$bias_synapse_update, function(x){x * model$momentum})
  
  # model$time_synapse_ouput_update = model$time_synapse_ouput_update * model$momentum
  # model$bias_synapse_ouput_update = model$bias_synapse_ouput_update * model$momentum
  # 
  # if(!is.null(model$clipping)){
  #   clipping = function(x){
  #     x[x > model$clipping] = model$clipping
  #     x[x < -model$clipping] = - model$clipping
  #     return(x)
  #   }
  #   model$recurrent_synapse  = lapply(model$recurrent_synapse,clipping)
  #   model$time_synapse       = lapply(model$time_synapse,clipping)
  #   # model$bias_synapse       = lapply(model$bias_synapse, clipping)
  #   model$time_synapse_ouput = clipping(model$time_synapse_ouput)
  #   # model$bias_synapse_ouput = clipping(model$bias_synapse_ouput)
  # }
  
  
  # 
  # model$synapse_0_i = model$synapse_0_i + model$synapse_0_i_update
  # model$synapse_0_f = model$synapse_0_f + model$synapse_0_f_update
  # model$synapse_0_o = model$synapse_0_o + model$synapse_0_o_update
  # model$synapse_0_c = model$synapse_0_c + model$synapse_0_c_update
  # model$synapse_1   = model$synapse_1   + model$synapse_1_update  
  # model$synapse_h_i = model$synapse_h_i + model$synapse_h_i_update
  # model$synapse_h_f = model$synapse_h_f + model$synapse_h_f_update
  # model$synapse_h_o = model$synapse_h_o + model$synapse_h_o_update
  # model$synapse_h_c = model$synapse_h_c + model$synapse_h_c_update
  # model$synapse_b_1   = model$synapse_b_1   + model$synapse_b_1_update  
  # model$synapse_b_i = model$synapse_b_i + model$synapse_b_i_update
  # model$synapse_b_f = model$synapse_b_f + model$synapse_b_f_update
  # model$synapse_b_o = model$synapse_b_o + model$synapse_b_o_update
  # model$synapse_b_c = model$synapse_b_c + model$synapse_b_c_update
  # 
  # model$synapse_0_i_update = model$synapse_0_i_update* model$momentum
  # model$synapse_0_f_update = model$synapse_0_f_update* model$momentum
  # model$synapse_0_o_update = model$synapse_0_o_update* model$momentum
  # model$synapse_0_c_update = model$synapse_0_c_update* model$momentum
  # model$synapse_1_update   = model$synapse_1_update  * model$momentum
  # model$synapse_h_i_update = model$synapse_h_i_update* model$momentum
  # model$synapse_h_f_update = model$synapse_h_f_update* model$momentum
  # model$synapse_h_o_update = model$synapse_h_o_update* model$momentum
  # model$synapse_h_c_update = model$synapse_h_c_update* model$momentum
  # model$synapse_b_1_update = model$synapse_b_1_update* model$momentum
  # model$synapse_b_i_update = model$synapse_b_i_update* model$momentum
  # model$synapse_b_f_update = model$synapse_b_f_update* model$momentum
  # model$synapse_b_o_update = model$synapse_b_o_update* model$momentum
  # model$synapse_b_c_update = model$synapse_b_c_update* model$momentum
  

  
  
  return(model)
}

