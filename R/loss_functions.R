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
    # model$recurrent_synapse_update  = lapply(model$recurrent_synapse_update,function(x){x * model$learningrate})
    # model$time_synapse_update       = lapply(model$time_synapse_update,function(x){x * model$learningrate})
    # model$bias_synapse_update       = lapply(model$bias_synapse_update, function(x){x * model$learningrate})
    # model$time_synapse_ouput_update = model$time_synapse_ouput_update * model$learningrate
    # model$bias_synapse_ouput_update = model$bias_synapse_ouput_update * model$learningrate
    
    model$synapse_1_update   = model$synapse_1_update   * model$learningrate
    model$synapse_h_i_update = model$synapse_h_i_update * model$learningrate
    model$synapse_h_f_update = model$synapse_h_f_update * model$learningrate
    model$synapse_h_o_update = model$synapse_h_o_update * model$learningrate
    model$synapse_h_c_update = model$synapse_h_c_update * model$learningrate
    model$synapse_0_i_update = model$synapse_0_i_update * model$learningrate
    model$synapse_0_f_update = model$synapse_0_f_update * model$learningrate
    model$synapse_0_o_update = model$synapse_0_o_update * model$learningrate
    model$synapse_0_c_update = model$synapse_0_c_update * model$learningrate
    model$synapse_b_1_update = model$synapse_b_1_update * model$learningrate
    model$synapse_b_i_update = model$synapse_b_i_update * model$learningrate
    model$synapse_b_f_update = model$synapse_b_f_update * model$learningrate
    model$synapse_b_o_update = model$synapse_b_o_update * model$learningrate
    model$synapse_b_c_update = model$synapse_b_c_update * model$learningrate
    
  }
  return(model)
}

