#' @name update_r
#' @title update_r
#' @description Apply the update
#' @param model the output model object
#' @return the updated model

update_r = function(model){
  if(model$update_rule == "sgd"){
    update_sgd(model)
  }else if(model$update_rule == "adagrad"){
    update_adagrad(model)
  }else{
    stop("update_rule unknown")
  }
}

#' @name update_sgd
#' @title update_sgd
#' @description Apply the update with stochastic gradient descent
#' @param model the output model object
#' @return the updated model

update_sgd = function(model){
  
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
  
  for(i in seq(length(model$time_synapse))){
    model$time_synapse[[i]] <- model$time_synapse[[i]] + model$time_synapse_update[[i]]
    model$bias_synapse[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
  }
  for(i in seq(length(model$recurrent_synapse))){
    model$recurrent_synapse[[i]] <- model$recurrent_synapse[[i]] + model$recurrent_synapse_update[[i]]
  }
  
  # Initializing the update with the momentum
  model$time_synapse_update = lapply(model$time_synapse_update,function(x){x* model$momentum})
  model$bias_synapse_update = lapply(model$bias_synapse_update,function(x){x* model$momentum})
  model$recurrent_synapse_update = lapply(model$recurrent_synapse_update,function(x){x* model$momentum})
  
  return(model)
}


#' @name update_adagrad
#' @title update_adagrad
#' @description Apply the update with adagrad, not working yet
#' @param model the output model object
#' @return the updated model

update_adagrad = function(model){
  ## not working yet, inspiration here:
  ## https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
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
  
  if(is.null(model$recurrent_synapse_update_old)){ # not really the old update but just a store of the old
    model$recurrent_synapse_update_old = lapply(model$recurrent_synapse_update,function(x){x*0})
    model$time_synapse_update_old = lapply(model$time_synapse_update,function(x){x*0})
    # model$bias_synapse_update_old = lapply(model$bias_synapse_update,function(x){x*0}) # the bias stay the same, we only apply it on the weight
  }
  
  for(i in seq(length(model$time_synapse))){
    model$time_synapse_update_old[[i]] <- model$time_synapse_update_old[[i]] + model$time_synapse_update[[i]]
    # model$bias_synapse_old[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
  }
  for(i in seq(length(model$recurrent_synapse))){
    model$recurrent_synapse_update_old[[i]] <- model$recurrent_synapse_update_old[[i]] + model$recurrent_synapse_update[[i]]
  }
  
  for(i in seq(length(model$time_synapse))){
    model$time_synapse[[i]] <- model$time_synapse[[i]] + model$learningrate * model$time_synapse_update[[i]] / (model$time_synapse_update_old[[i]] + 0.000000001)
    model$bias_synapse[[i]] <- model$bias_synapse[[i]] + model$bias_synapse_update[[i]]
  }
  for(i in seq(length(model$recurrent_synapse))){
    model$recurrent_synapse[[i]] <- model$recurrent_synapse[[i]] + model$learningrate * model$recurrent_synapse_update[[i]] / (model$recurrent_synapse_update_old[[i]] + 0.000000001)
  }
  
  return(model)
}