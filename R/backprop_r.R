#' @name backprop_r
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
  } else if (model$network_type == "gru"){
    backprop_gru(model,a,c,j,...)
  }else{
    stop("network_type_unknown for the backprop")
  }
}

#' @name backprop_rnn
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
  
  if(model$seq_to_seq_unsync){
    model$last_layer_error[j,1:(model$time_dim_input - 1),] = 0
    model$last_layer_delta[j,1:(model$time_dim_input - 1),] = 0
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
#' @title backprop_lstm
#' @description backpropagate the error in a model object of type rlstm
#' @importFrom sigmoid tanh_output_to_derivative
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
  
  if(model$seq_to_seq_unsync){
    model$last_layer_error[j,1:(model$time_dim_input - 1),] = 0
    model$last_layer_delta[j,1:(model$time_dim_input - 1),] = 0
  }
  model$error[j,model$current_epoch] <- apply(model$last_layer_error[j,,,drop=F],1,function(x){sum(abs(x))})
  
  future_layer_cell_delta = list()
  future_layer_hidden_delta = list()
  for(i in seq(length(model$hidden_dim))){
    future_layer_cell_delta[[i]] = matrix(0, nrow = length(j), ncol = model$hidden_dim[i]) # 4 to actualize
    future_layer_hidden_delta[[i]] = matrix(0, nrow = length(j), ncol = model$hidden_dim[i]) # 2, to actualize
  }
  
  
  for (position in model$time_dim:1) {
    # error at output layer
    layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
    
    # first the last layer to update the layer_up_delta
    i = length(model$hidden_dim)
    layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
    # output layer update
    model$time_synapse_update[[i+1]]   = model$time_synapse_update[[i+1]]   + (t(layer_hidden) %*% layer_up_delta)
    model$bias_synapse_update[[i+1]]   = model$bias_synapse_update[[i+1]]   + colMeans(layer_up_delta)
    # lstm hidden delta
    layer_up_delta = (layer_up_delta %*% t(model$time_synapse_update[[i+1]])) * sigmoid_output_to_derivative(layer_hidden) + future_layer_hidden_delta[[i]] # 1 and 3
    
    for(i in length(model$hidden_dim):1){
      # x: input of the layer
      if(i == 1){
        x = array(a[,position,],dim=c(length(j),model$input_dim))
      }else{
        x = array(model$store[[i - 1]][j,position,,1],dim=c(length(j),model$synapse_dim[i]))
      }
      layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
      layer_cell = array(model$store[[i]][j,position,,2],dim=c(length(j), model$hidden_dim[i]))
      if(position != 1){
        prev_layer_hidden =array(model$store[[i]][j,position-1,,1],dim=c(length(j),model$hidden_dim[i]))
        preview_layer_cell = array(model$store[[i]][j,position-1,,2],dim=c(length(j), model$hidden_dim[i]))
      }else{
        prev_layer_hidden =array(0,dim=c(length(j),model$hidden_dim[i]))
        preview_layer_cell = array(0,dim=c(length(j), model$hidden_dim[i]))
      }
      
      layer_f = array(model$store[[i]][j,position,,3],dim=c(length(j), model$hidden_dim[i]))
      layer_i = array(model$store[[i]][j,position,,4],dim=c(length(j), model$hidden_dim[i]))
      layer_c = array(model$store[[i]][j,position,,5],dim=c(length(j), model$hidden_dim[i]))
      layer_o = array(model$store[[i]][j,position,,6],dim=c(length(j), model$hidden_dim[i]))
      
      # lstm cell delta
      # layer_cell_delta = (layer_hidden_delta * layer_o) + future_layer_cell_delta    # 5 then 8 (skip 7 as no tanh)
      # layer_o_delta_post_activation = layer_hidden_delta *  layer_cell # 6 (skip 7 as no tanh)
      layer_cell_delta = (layer_up_delta * layer_o)* tanh_output_to_derivative(layer_cell) + future_layer_cell_delta[[i]]    # 5, 7 then 8
      layer_o_delta_post_activation = layer_up_delta *  tanh(layer_cell) # 6 
      
      layer_c_delta_post_activation = layer_cell_delta * layer_i    # 9
      layer_i_delta_post_activation = layer_cell_delta * layer_c    # 10
      
      layer_f_delta_post_activation = layer_cell_delta * preview_layer_cell # 12
      future_layer_cell_delta[[i]] = layer_cell_delta * layer_f # 11
      
      layer_o_delta_pre_activation = layer_o_delta_post_activation * sigmoid_output_to_derivative(layer_o) # 13
      layer_c_delta_pre_activation = layer_c_delta_post_activation * tanh_output_to_derivative(layer_c) # 14
      layer_i_delta_pre_activation = layer_i_delta_post_activation * sigmoid_output_to_derivative(layer_i) # 15
      layer_f_delta_pre_activation = layer_f_delta_post_activation * sigmoid_output_to_derivative(layer_f) # 16
      # 
      
      
      # let's update all our weights so we can try again
      model$recurrent_synapse_update[[i]][,,1] = model$recurrent_synapse_update[[i]][,,1] + t(prev_layer_hidden) %*% layer_f_delta_post_activation
      model$recurrent_synapse_update[[i]][,,2] = model$recurrent_synapse_update[[i]][,,2] + t(prev_layer_hidden) %*% layer_i_delta_post_activation
      model$recurrent_synapse_update[[i]][,,3] = model$recurrent_synapse_update[[i]][,,3] + t(prev_layer_hidden) %*% layer_c_delta_post_activation
      model$recurrent_synapse_update[[i]][,,4] = model$recurrent_synapse_update[[i]][,,4] + t(prev_layer_hidden) %*% layer_o_delta_post_activation
      model$time_synapse_update[[i]][,,1] = model$time_synapse_update[[i]][,,1] + t(x) %*% layer_f_delta_post_activation
      model$time_synapse_update[[i]][,,2] = model$time_synapse_update[[i]][,,2] + t(x) %*% layer_i_delta_post_activation
      model$time_synapse_update[[i]][,,3] = model$time_synapse_update[[i]][,,3] + t(x) %*% layer_c_delta_post_activation
      model$time_synapse_update[[i]][,,4] = model$time_synapse_update[[i]][,,4] + t(x) %*% layer_o_delta_post_activation
      model$bias_synapse_update[[i]][,1] = model$bias_synapse_update[[i]][,1] + colMeans(layer_f_delta_post_activation)
      model$bias_synapse_update[[i]][,2] = model$bias_synapse_update[[i]][,2] + colMeans(layer_i_delta_post_activation)
      model$bias_synapse_update[[i]][,3] = model$bias_synapse_update[[i]][,3] + colMeans(layer_c_delta_post_activation)
      model$bias_synapse_update[[i]][,4] = model$bias_synapse_update[[i]][,4] + colMeans(layer_o_delta_post_activation)
      
      layer_f_delta_pre_weight = layer_f_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,1],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 20
      layer_i_delta_pre_weight = layer_i_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,2],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 19
      layer_c_delta_pre_weight = layer_c_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,3],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 18
      layer_o_delta_pre_weight = layer_o_delta_pre_activation %*% t(array(model$recurrent_synapse[[i]][,,4],dim=c(dim(model$recurrent_synapse[[i]])[1:2]))) # 17
      future_layer_hidden_delta[[i]] = layer_o_delta_pre_weight + layer_c_delta_pre_weight + layer_i_delta_pre_weight + layer_f_delta_pre_weight # 21
      
      layer_f_delta_pre_weight = layer_f_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,1],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 20
      layer_i_delta_pre_weight = layer_i_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,2],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 19
      layer_c_delta_pre_weight = layer_c_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,3],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 18
      layer_o_delta_pre_weight = layer_o_delta_pre_activation %*% t(array(model$time_synapse[[i]][,,4],dim=c(dim(model$time_synapse[[i]])[1:2]))) # 17
      layer_up_delta = layer_o_delta_pre_weight + layer_c_delta_pre_weight + layer_i_delta_pre_weight + layer_f_delta_pre_weight # 21
    }
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

#' @name backprop_gru
#' @title backprop_gru
#' @description backpropagate the error in a model object of type gru
#' @importFrom sigmoid tanh_output_to_derivative
#' @param model the output model object
#' @param a the input of this learning batch
#' @param c the output of this learning batch
#' @param j the indexes of the sample in the current batch
#' @param ... argument to be passed to method
#' @return the updated model

backprop_gru = function(model,a,c,j,...){
  
  # store errors
  model$last_layer_error[j,,] = c - model$store[[length(model$store)]][j,,,drop=F]
  model$last_layer_delta[j,,] = model$last_layer_error[j,,,drop = F] * sigmoid_output_to_derivative(model$store[[length(model$store)]][j,,,drop=F])
  
  # many_to_one
  if(model$seq_to_seq_unsync){
    model$last_layer_error[j,1:(model$time_dim_input - 1),] = 0
    model$last_layer_delta[j,1:(model$time_dim_input - 1),] = 0
  }
  model$error[j,model$current_epoch] <- apply(model$last_layer_error[j,,,drop=F],1,function(x){sum(abs(x))})
  
  future_layer_hidden_delta = list()
  for(i in seq(length(model$hidden_dim))){
    future_layer_hidden_delta[[i]] = matrix(0, nrow = length(j), ncol = model$hidden_dim[i]) # 2, to actualize
  }
  
  
  for (position in model$time_dim:1) {
    # error at output layer
    layer_up_delta = array(model$last_layer_delta[j,position,],dim=c(length(j),model$output_dim))
    
    # first the last layer to update the layer_up_delta
    i = length(model$hidden_dim)
    layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
    # output layer update
    model$time_synapse_update[[i+1]]   = model$time_synapse_update[[i+1]]   + (t(layer_hidden) %*% layer_up_delta)
    model$bias_synapse_update[[i+1]]   = model$bias_synapse_update[[i+1]]   + colMeans(layer_up_delta)
    # lstm hidden delta
    layer_up_delta = (layer_up_delta %*% t(model$time_synapse_update[[i+1]])) * sigmoid_output_to_derivative(layer_hidden) + future_layer_hidden_delta[[i]] # 1 and 3
    
    for(i in length(model$hidden_dim):1){
      # x: input of the layer
      if(i == 1){
        x = array(a[,position,],dim=c(length(j),model$input_dim))
      }else{
        x = array(model$store[[i - 1]][j,position,,1],dim=c(length(j),model$synapse_dim[i]))
      }
      layer_hidden = array(model$store[[i]][j,position,,1],dim=c(length(j),model$hidden_dim[i]))
      if(position != 1){
        prev_layer_hidden =array(model$store[[i]][j,position-1,,1],dim=c(length(j),model$hidden_dim[i]))
      }else{
        prev_layer_hidden =array(0,dim=c(length(j),model$hidden_dim[i]))
      }
      
      layer_z = array(model$store[[i]][j,position,,2],dim=c(length(j), model$hidden_dim[i]))
      layer_r = array(model$store[[i]][j,position,,3],dim=c(length(j), model$hidden_dim[i]))
      layer_h = array(model$store[[i]][j,position,,4],dim=c(length(j), model$hidden_dim[i]))
      
      layer_hidden_delta = layer_up_delta + future_layer_hidden_delta[[i]] #3
      layer_h_delta_post_activation = layer_hidden_delta *  layer_z # 6 
      layer_h_delta_pre_activation = layer_h_delta_post_activation * tanh_output_to_derivative(layer_h) # 6 bis
      layer_z_delta_post_split = layer_hidden_delta *  layer_h # 7 
      
      layer_z_delta_post_1_minus = layer_hidden_delta *  prev_layer_hidden # 9 
      layer_hidden_delta = layer_hidden_delta * (1 - layer_z) # 8
      
      layer_z_delta_post_activation = (1 - layer_z_delta_post_1_minus) # 10
      layer_z_delta_pre_activation = layer_z_delta_post_activation*  sigmoid_output_to_derivative(layer_z) # 10 bis
      layer_z_delta_pre_weight_h = (layer_z_delta_pre_activation %*% t(model$recurrent_synapse[[i]][,,1]) ) # 14 
      layer_z_delta_pre_weight_x = (layer_z_delta_pre_activation %*% array(t(model$time_synapse[[i]][,,1]),dim = dim(model$time_synapse[[i]])[2:1])) # 14 
      # let's update all our weights so we can try again
      model$recurrent_synapse_update[[i]][,,1] = model$recurrent_synapse_update[[i]][,,1] + t(prev_layer_hidden) %*% layer_z_delta_post_activation
      model$time_synapse_update[[i]][,,1] = model$time_synapse_update[[i]][,,1] + t(x) %*% layer_z_delta_post_activation
      model$bias_synapse_update[[i]][,1] = model$bias_synapse_update[[i]][,1] + colMeans(layer_z_delta_post_activation)
      
      layer_h_delta_pre_weight_h = (layer_h_delta_pre_activation %*% t(model$recurrent_synapse[[i]][,,3]))# 13 
      layer_h_delta_pre_weight_x = ( layer_h_delta_pre_activation %*% array(t(model$time_synapse[[i]][,,3]),dim = dim(model$time_synapse[[i]])[2:1])) # 13 
      # let's update all our weights so we can try again
      model$recurrent_synapse_update[[i]][,,3] = model$recurrent_synapse_update[[i]][,,3] + t(prev_layer_hidden * layer_r) %*% layer_h_delta_post_activation
      model$time_synapse_update[[i]][,,3] = model$time_synapse_update[[i]][,,3] + t(x) %*% layer_h_delta_post_activation
      model$bias_synapse_update[[i]][,3] = model$bias_synapse_update[[i]][,3] + colMeans(layer_h_delta_post_activation)

      layer_r_delta_post_activation = prev_layer_hidden * layer_h_delta_pre_weight_h # 15
      layer_r_delta_pre_activation = layer_r_delta_post_activation * sigmoid_output_to_derivative(layer_r) # 15 bis
      layer_hidden_delta = layer_hidden_delta + layer_r * layer_h_delta_pre_weight_h # 12
      
      layer_r_delta_pre_weight_h = (layer_r_delta_pre_activation %*% t(model$recurrent_synapse[[i]][,,2])) # 17 
      layer_r_delta_pre_weight_x = (layer_r_delta_post_activation %*% array(t(model$time_synapse[[i]][,,2]),dim = dim(model$time_synapse[[i]])[2:1])) # 17 
      # let's update all our weights so we can try again
      model$recurrent_synapse_update[[i]][,,2] = model$recurrent_synapse_update[[i]][,,2] + t(prev_layer_hidden) %*% layer_r_delta_post_activation
      model$time_synapse_update[[i]][,,2] = model$time_synapse_update[[i]][,,2] + t(x) %*% layer_r_delta_post_activation
      model$bias_synapse_update[[i]][,2] = model$bias_synapse_update[[i]][,2] + colMeans(layer_r_delta_post_activation)
      
      layer_r_and_z_delta_pre_weight_h = layer_r_delta_pre_weight_h + layer_z_delta_pre_weight_h # 19
      layer_r_and_z_delta_pre_weight_x = layer_r_delta_pre_weight_x + layer_z_delta_pre_weight_x # 19
      
      future_layer_hidden_delta[[i]] = layer_hidden_delta + layer_r_and_z_delta_pre_weight_h # 23
      
      layer_up_delta = layer_r_and_z_delta_pre_weight_x + layer_h_delta_pre_weight_x # 22
    }
  }
  return(model)
}


