#' @name epoch_print
#' @export
#' @title epoch printing for trainr
#' @description Print the error adn learning rate at each epoch of the trainr learning, called in epoch_function
#' @param model the output model object
#' @return nothing

epoch_print = function(model){
  message(paste0("Trained epoch: ",model$current_epoch," - Learning rate: ",model$learningrate))
  message(paste0("Epoch error: ",colMeans(model$error)[model$current_epoch]))
  return(model)
}

#' @name epoch_annealing
#' @export
#' @title epoch annealing
#' @description Apply the learning rate decay to the learning rate, called in epoch_model_function
#' @param model the output model object
#' @return the updated model

epoch_annealing = function(model){
  model$learningrate = model$learningrate * model$learningrate_decay
  return(model)
}

