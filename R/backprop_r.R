#' @name backprop_r
#' @export
#' @title backprop_r
#' @description backpropagate the error in a model object
#' @param model the output model object
#' @param a the input of this learning batch, only one sample for the moment so batch is a little bit overrated
#' @param ... argument to be passed to method
#' @return the updated model

backprop_r = function(model,a,c,j,...){
  if(model$network_type == "rnn"){
    backprop_rnn(model,a,c,j,...)
  }else{
    stop("only rnn supported for the moment")
  }
}

#' @name backprop_rnn
#' @export
#' @title backprop_rnn
#' @description backpropagate the error in a model object of type rnn
#' @param model the output model object
#' @param a the input of this learning batch, only one sample for the moment so batch is a little bit overrated
#' @param ... argument to be passed to method
#' @return the updated model

backprop_rnn = function(model,a,c,j,...){

  # store errors
  model$last_layer_error[j,,] = c - model$store[[length(model$store)]][j,,,drop=F]
  model$last_layer_delta[j,,] = model$last_layer_error[j,,,drop = F] * sigmoid_output_to_derivative(model$store[[length(model$store)]][j,,,drop=F])
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
