#' @name trainr
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid sigmoid_output_to_derivative
#' @title Recurrent Neural Network
#' @description Trains a Recurrent Neural Network.
#' @param Y array of output values, dim 1: samples (must be equal to dim 1 of X), dim 2: time (must be equal to dim 2 of X), dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param learningrate learning rate to be applied for weight iteration
#' @param numepochs number of iteration, i.e. number of time the whole dataset is presented to the network
#' @param hidden_dim dimension(s) of hidden layer(s)
#' @param network_type type of network, could be rnn or lstm, only rnn supported for the moment
#' @param batch_size batch size: number of samples used at each weight iteration, only 1 supported for the moment
#' @param sigmoid method to be passed to the sigmoid function
#' @param start_from_end should the sequence start from the end, legacy of the binary example. DEPRECATED, first index is the begining
#' @param learningrate_decay coefficient to apply to the learning rate at each epoch, via the epoch_annealing function
#' @param momentum coefficient of the last weight iteration to keep for faster learning
#' @param use_bias should the network use bias
#' @param epoch_function vector of functions with no side effect on the model to applied at each epoch loop
#' @param epoch_model_function vector of functions with side effect on the model to applied at each epoch loop
#' @param loss_function loss function, applied in each sample loop, vocabulary to verify
#' @param ... Arguments to be passed to methods, to be used in user defined functions
#' @return a model to be used by the predictr function
#' @examples 
#' # create training numbers
#' X1 = sample(0:127, 7000, replace=TRUE)
#' X2 = sample(0:127, 7000, replace=TRUE)
#' 
#' # create training response numbers
#' Y <- X1 + X2
#' 
#' # convert to binary
#' X1 <- int2bin(X1, length=8)
#' X2 <- int2bin(X2, length=8)
#' Y  <- int2bin(Y,  length=8)
#' 
#' # create 3d array: dim 1: samples; dim 2: time; dim 3: variables
#' X <- array( c(X1,X2), dim=c(dim(X1),2) )
#' 
#' # train the model
#' model <- trainr(Y=Y,
#'                 X=X,
#'                 learningrate   =  0.1,
#'                 hidden_dim     = 10,
#'                 start_from_end = TRUE )
#'     

trainr <- function(Y, X, learningrate, learningrate_decay = 1, momentum = 0, hidden_dim = c(10),network_type = "rnn",
                   numepochs = 1, sigmoid = c('logistic', 'Gompertz', 'tanh'), start_from_end=FALSE, use_bias = F, batch_size = 1,
                   epoch_function = c(epoch_print),
                   epoch_model_function = c(epoch_annealing),
                   loss_function = loss_L1,...) {
  
  #  find sigmoid
  sigmoid <- match.arg(sigmoid)
  
  # check the consistency
  if(dim(X)[2] != dim(Y)[2]){
    stop("The time dimension of X is different from the time dimension of Y. Only sequences to sequences is supported")
  }
  if(dim(X)[1] != dim(Y)[1]){
    stop("The sample dimension of X is different from the sample dimension of Y.")
  }
  
  # coerce to array if matrix
  if(length(dim(X)) == 2){
    X <- array(X,dim=c(dim(X),1))
  }
  if(length(dim(Y)) == 2){
    Y <- array(Y,dim=c(dim(Y),1))
  }
  
  # reverse the time dim if start from end
  # if(start_from_end){
  #   X <- X[,dim(X)[2]:1,,drop = F]
  #   Y <- Y[,dim(X)[2]:1,,drop = F]
  # }
  
  # initialize the model list
  model                         = list(...) # we start by the ... argument before appending everybody else
  model$input_dim               = dim(X)[3]
  model$hidden_dim              = hidden_dim
  model$output_dim              = dim(Y)[3]
  model$synapse_dim             = c(model$input_dim,model$hidden_dim,model$output_dim)
  model$time_dim                = dim(X)[2] ## changed from binary_dim to get rid of the binary user case legacy
  model$sigmoid                 = sigmoid
  model$network_type            = network_type
  model$numepochs               = numepochs
  model$batch_size               = batch_size
  model$learningrate            = learningrate
  model$learningrate_decay      = learningrate_decay ## this one should be in the ... arg and be here initially but he was supply before
  model$momentum                = momentum
  model$use_bias                = use_bias
  model$start_from_end          = start_from_end
  model$epoch_function          = epoch_function
  model$epoch_model_function    = epoch_model_function
  model$loss_function           = loss_function
  model$last_layer_error        = Y*0
  model$last_layer_delta        = Y*0
  
  # Storing layers states, filled with 0 for the moment
  model$store <- list()
  for(i in seq(length(model$synapse_dim) - 1)){
    model$store[[i]] <- array(0,dim = c(dim(Y)[1:2],model$synapse_dim[i+1]))
  }
  
  if(model$network_type == "rnn"){
    model <- init_rnn(model)
  }else{
    stop("only rnn supported for the moment")
  }
  
  
  # Storing errors, dim 1: samples, dim 2 is epochs, we could store also the time and variable dimension
  model$error <- array(0,dim = c(dim(Y)[1],model$numepochs))
  
  # training logic
  for(epoch in seq(model$numepochs)){
    model$current_epoch = epoch
    index = sample(seq(round(dim(Y)[1]/model$batch_size)),dim(Y)[1],replace = T)
    lj = list()
    for(i in seq(round(dim(Y)[1]/model$batch_size))){lj[[i]] = seq(dim(Y)[1])[index == i]}
    for (j in lj) {
    # for (j in 1:dim(Y)[1]) {
      # generate input and output for the sample loop
      a = X[j,,,drop=F]
      c = Y[j,,,drop=F]
      
      # feed forward
      store = predictr(model,a,hidden = T,real_output = F)
      for(i in seq(length(model$synapse_dim) - 1)){
        model$store[[i]][j,,] = store[[i]]
      }
      
      # apply back propagation
      model = backprop_r(model,a,c,j)
      
      # apply the loss function, default is to apply L1 learning rate, vocabulary to verify.
      model = model$loss_function(model)
      
      # Applying the update
      model = update_r(model)
      
    } # end sample loop
    
    # update best guess if error is minimal, will make more sens to store the weight...
    if(colMeans(model$error)[epoch] <= min(colMeans(model$error)[1:epoch])){
      model$store_best <- model$store
    }
    
    # epoch_function
    for(i in model$epoch_function){
      i(model)
    }
    
    # # epoch_model_function
    for(i in model$epoch_model_function){
      model <- i(model)
    }
    
  } # end epoch loop
  
  # clean model object, get rid of the update mainly, potentially other cleaning if not necessary in predictr
  model = clean_r(model)
  
  attr(model, 'error') <- colMeans(model$error)
  
  # return output
  return(model)
  
}
