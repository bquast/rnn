#' @name backprop_r
#' @export
#' @title backprop_r
#' @description backpropagate the error in a model object
#' @param model the output model object
#' @param a the input of this learning batch
#' @param c the output of this learning batch
#' @param j the indexes of the sample in the current batch
#' @param ... argument to be passed to method
#' @return the updated model

backprop_r = function(model,a,c,j,...){
  if(model$network_type == "rnn"){
    backprop_rnn(model,a,c,j,...)
  } else if (model$network_type == "lstm"){
    backprop_lstm(model,a,c,j,...)
  }else{
    stop("network_type_unknown for the backprop")
  }
}

#' @name backprop_rnn
#' @export
#' @title backprop_rnn
#' @description backpropagate the error in a model object of type rnn
#' @param model the output model object
#' @param a the input of this learning batch
#' @param c the output of this learning batch
#' @param j the indexes of the sample in the current batch
#' @param ... argument to be passed to method
#' @return the updated model

backprop_rnn = function(model,a,c,j,...){

  # store errors
  model$last_layer_error[j,,] = c - model$store[[length(model$store)]][j,,,drop=F]
  model$last_layer_delta[j,,] = model$last_layer_error[j,,,drop = F] * sigmoid_output_to_derivative(model$store[[length(model$store)]][j,,,drop=F])
  
  # many_to_one
  if(model$many_to_one){
    model$last_layer_error[j,1:(model$time_dim - 1),] = 0
    model$last_layer_delta[j,1:(model$time_dim - 1),] = 0
  }
  
  model$error[j,model$current_epoch] <- apply(model$last_layer_error[j,,,drop=F],1,function(x){sum(abs(x))})
  
  # init futur layer delta, here because there is no layer delta at time_dim+1
  future_layer_delta  = list()
  for(i in seq(length(model$hidden_dim))){
    future_layer_delta[[i]] <- matrix(0,nrow=length(j), ncol = model$hidden_dim[i])
  }
  
  # Weight iteration,
  for (position in model$time_dim:1) {
    
    # input states
    x            = array(a[,position,],dim=c(length(j),model$input_dim))
    # error at output layer
    layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))

    for(i in (length(model$store)):1){
      if(i != 1){ # need update for time and recurrent synapse
        layer_current      = array(model$store[[i-1]][j,position,],dim=c(length(j),model$hidden_dim[i-1]))
        if(position != 1){
          prev_layer_current = array(model$store[[i-1]][j,position - 1,],dim=c(length(j),model$hidden_dim[i-1]))
        }else{
          prev_layer_current = array(0,dim=c(length(j),model$hidden_dim[i-1]))
        }
        # error at hidden layers
        layer_current_delta = (future_layer_delta[[i-1]] %*% t(model$recurrent_synapse[[i-1]]) + layer_up_delta %*% t(model$time_synapse[[i]])) *
          sigmoid_output_to_derivative(layer_current)
        model$time_synapse_update[[i]] = model$time_synapse_update[[i]] + t(layer_current) %*% layer_up_delta
        model$bias_synapse_update[[i]] = model$bias_synapse_update[[i]] + colMeans(layer_up_delta)
        model$recurrent_synapse_update[[i-1]] = model$recurrent_synapse_update[[i-1]] + t(prev_layer_current) %*% layer_current_delta
        layer_up_delta = layer_current_delta
        future_layer_delta[[i-1]] = layer_current_delta
      }else{ # need only update for time synapse
        model$time_synapse_update[[i]] = model$time_synapse_update[[i]] + t(x) %*% layer_up_delta
      }
    }
  } # end position back prop loop
  return(model)
}

#' @name backprop_lstm
#' @export
#' @title backprop_lstm
#' @description backpropagate the error in a model object of type rlstm
#' @param model the output model object
#' @param a the input of this learning batch
#' @param c the output of this learning batch
#' @param j the indexes of the sample in the current batch
#' @param ... argument to be passed to method
#' @return the updated model

backprop_lstm = function(model,a,c,j,...){
  
  # store errors
  model$last_layer_error[j,,] = c - model$store[[length(model$store)]][j,,,drop=F]
  model$last_layer_delta[j,,] = model$last_layer_error[j,,,drop = F] * sigmoid_output_to_derivative(model$store[[length(model$store)]][j,,,drop=F])
  
  # many_to_one
  if(model$many_to_one){
    model$last_layer_error[j,1:(model$time_dim - 1),] = 0
    model$last_layer_delta[j,1:(model$time_dim - 1),] = 0
  }
  
  model$error[j,model$current_epoch] <- apply(model$last_layer_error[j,,,drop=F],1,function(x){sum(abs(x))})
  
  future_layer_1_i_delta = matrix(0, nrow = length(j), ncol = model$hidden_dim)
  future_layer_1_f_delta = matrix(0, nrow = length(j), ncol = model$hidden_dim)
  future_layer_1_o_delta = matrix(0, nrow = length(j), ncol = model$hidden_dim)
  future_layer_1_c_delta = matrix(0, nrow = length(j), ncol = model$hidden_dim)
  
  for (position in model$time_dim:1) {
    
    # input states
    x            = array(a[,position,],dim=c(length(j),model$input_dim))
    # error at output layer
    layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
    
    layer_1 = array(model$store[[1]][j,position,],dim=c(length(j),model$hidden_dim[1]))
    if(position != 1){
      prev_layer_1 =array(model$store[[1]][j,position-1,],dim=c(length(j),model$hidden_dim[1]))
    }else{
      prev_layer_1 =array(0,dim=c(length(j),model$hidden_dim[1]))
    }
    
    # error at hidden layer
    layer_1_i_delta = (future_layer_1_i_delta %*% t(model$synapse_h_i) + layer_up_delta %*% t(model$synapse_1)) *
      sigmoid_output_to_derivative(layer_1)
    layer_1_f_delta = (future_layer_1_f_delta %*% t(model$synapse_h_f) + layer_up_delta %*% t(model$synapse_1)) *
      sigmoid_output_to_derivative(layer_1)
    layer_1_o_delta = (future_layer_1_o_delta %*% t(model$synapse_h_o) + layer_up_delta %*% t(model$synapse_1)) *
      sigmoid_output_to_derivative(layer_1)
    layer_1_c_delta = (future_layer_1_c_delta %*% t(model$synapse_h_c) + layer_up_delta %*% t(model$synapse_1)) *
      sigmoid_output_to_derivative(layer_1)

    # let's update all our weights so we can try again
    model$synapse_1_update   = model$synapse_1_update   + t(layer_1)      %*% layer_up_delta
    model$synapse_h_i_update = model$synapse_h_i_update + t(prev_layer_1) %*% layer_1_i_delta
    model$synapse_h_f_update = model$synapse_h_f_update + t(prev_layer_1) %*% layer_1_f_delta
    model$synapse_h_o_update = model$synapse_h_o_update + t(prev_layer_1) %*% layer_1_o_delta
    model$synapse_h_c_update = model$synapse_h_c_update + t(prev_layer_1) %*% layer_1_c_delta
    model$synapse_0_i_update = model$synapse_0_i_update + t(x) %*% layer_1_i_delta
    model$synapse_0_f_update = model$synapse_0_f_update + t(x) %*% layer_1_f_delta
    model$synapse_0_o_update = model$synapse_0_o_update + t(x) %*% layer_1_o_delta
    model$synapse_0_c_update = model$synapse_0_c_update + t(x) %*% layer_1_c_delta
    model$synapse_b_1_update = model$synapse_b_1_update + colMeans(layer_up_delta)
    model$synapse_b_i_update = model$synapse_b_i_update + colMeans(layer_1_i_delta)
    model$synapse_b_f_update = model$synapse_b_f_update + colMeans(layer_1_f_delta)
    model$synapse_b_o_update = model$synapse_b_o_update + colMeans(layer_1_o_delta)
    model$synapse_b_c_update = model$synapse_b_c_update + colMeans(layer_1_c_delta)
    
    future_layer_1_i_delta = layer_1_i_delta
    future_layer_1_f_delta = layer_1_f_delta
    future_layer_1_o_delta = layer_1_o_delta
    future_layer_1_c_delta = layer_1_c_delta
  }
  
  # future_layers_delta = list()
  # for(i in seq(length(model$hidden_dim))){
  #   future_layers_delta[[i]] = array(0,dim=c(length(j),model$hidden_dim[i],4))
  # }
  # 
  # for (position in model$time_dim:1) {
  #   
  #   # input states
  #   x            = array(a[,position,],dim=c(length(j),model$input_dim))
  #   # error at output layer
  #   layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
  #   
  #   for(i in (length(model$store)):1){
  #     if(i != 1){ # need update for time and recurrent synapse
  #       layer_current      = array(model$store[[i-1]][j,position,],dim=c(length(j),model$hidden_dim[i-1]))
  #       if(position != 1){
  #         prev_layer_current = array(model$store[[i-1]][j,position - 1,],dim=c(length(j),model$hidden_dim[i-1]))
  #       }else{
  #         prev_layer_current = array(0,dim=c(length(j),model$hidden_dim[i-1]))
  #       }
  #     }
  #     if(i == length(model$store)){
  #       # error at hidden layer
  #       future_layers_delta[[i-1]][,,1] = (future_layers_delta[[i-1]][,,1] %*% t(model$recurrent_synapse[[i-1]][,,1]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
  #         sigmoid_output_to_derivative(layer_current)
  #       future_layers_delta[[i-1]][,,2] = (future_layers_delta[[i-1]][,,2] %*% t(model$recurrent_synapse[[i-1]][,,2]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
  #         sigmoid_output_to_derivative(layer_current)
  #       future_layers_delta[[i-1]][,,3] = (future_layers_delta[[i-1]][,,3] %*% t(model$recurrent_synapse[[i-1]][,,3]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
  #         sigmoid_output_to_derivative(layer_current)
  #       future_layers_delta[[i-1]][,,4] = (future_layers_delta[[i-1]][,,4] %*% t(model$recurrent_synapse[[i-1]][,,4]) + layer_up_delta %*% t(model$time_synapse_ouput)) *
  #         sigmoid_output_to_derivative(layer_current)
  #       
  #       # let's update all our weights so we can try again
  #       model$time_synapse_ouput_update   = model$time_synapse_ouput_update   + t(layer_current)      %*% layer_up_delta
  #       model$recurrent_synapse_update[[i-1]][,,1]  = model$recurrent_synapse_update[[i-1]][,,1] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,1]
  #       model$recurrent_synapse_update[[i-1]][,,2]  = model$recurrent_synapse_update[[i-1]][,,2] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,2]
  #       model$recurrent_synapse_update[[i-1]][,,3]  = model$recurrent_synapse_update[[i-1]][,,3] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,3]
  #       model$recurrent_synapse_update[[i-1]][,,4]  = model$recurrent_synapse_update[[i-1]][,,4] + t(prev_layer_current) %*% future_layers_delta[[i-1]][,,4]
  #       
  #       model$bias_synapse_ouput_update = model$bias_synapse_ouput_update + colMeans(layer_up_delta)
  # 
  #       model$bias_synapse_update[[i-1]] = model$bias_synapse_update[[i-1]] + apply(future_layers_delta[[i-1]],2:3,mean)
  #       
  #     } else if(i == 1){
  #       model$time_synapse_update[[i]][,,1] = model$time_synapse_update[[i]][,,1] + t(x) %*% future_layers_delta[[i]][,,1]
  #       model$time_synapse_update[[i]][,,2] = model$time_synapse_update[[i]][,,2] + t(x) %*% future_layers_delta[[i]][,,2]
  #       model$time_synapse_update[[i]][,,3] = model$time_synapse_update[[i]][,,3] + t(x) %*% future_layers_delta[[i]][,,3]
  #       model$time_synapse_update[[i]][,,4] = model$time_synapse_update[[i]][,,4] + t(x) %*% future_layers_delta[[i]][,,4]
  #     }
  #   }
  # }
  return(model)
}

