#' @name trainr
#' @export
#' @importFrom stats runif
#' @importFrom sigmoid sigmoid sigmoid_output_to_derivative
#' @title Recurrent Neural Network
#' @description Trains a Recurrent Neural Network.
#' @param Y array of output values, dim 1: samples (must be equal to dim 1 of X), dim 2: time (must be equal to dim 2 of X), dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param X array of input values, dim 1: samples, dim 2: time, dim 3: variables (could be 1 or more, if a matrix, will be coerce to array)
#' @param model  a model trained before, used for retraining purpose.
#' @param learningrate learning rate to be applied for weight iteration
#' @param numepochs number of iteration, i.e. number of time the whole dataset is presented to the network
#' @param hidden_dim dimension(s) of hidden layer(s)
#' @param network_type type of network, could be rnn, gru or lstm. gru and lstm are experimentale.
#' @param batch_size batch size: number of samples used at each weight iteration, only 1 supported for the moment
#' @param sigmoid method to be passed to the sigmoid function
#' @param learningrate_decay coefficient to apply to the learning rate at each epoch, via the epoch_annealing function
#' @param momentum coefficient of the last weight iteration to keep for faster learning
#' @param update_rule rule to update the weight, "sgd", the default, is stochastic gradient descent, other available options are "adagrad" (experimentale, do not learn yet)
#' @param use_bias should the network use bias
#' @param seq_to_seq_unsync if TRUE, the network will be trained to backpropagate only the second half of the output error. If many to one is the target, just make Y have a time dim of 1. The X and Y data are modify at first to fit a classic learning, error are set to 0 during back propagation, input for the second part is also set to 0.
#' @param epoch_function vector of functions to applied at each epoch loop. Use it to intereact with the objects inside the list model or to print and plot at each epoch. Should return the model.
#' @param loss_function loss function, applied in each sample loop, vocabulary to verify.
#' @param ... Arguments to be passed to methods, to be used in user defined functions
#' @return a model to be used by the predictr function
#' @examples 
#' \dontrun{
#' # create training numbers
#' X1 = sample(0:127, 10000, replace=TRUE)
#' X2 = sample(0:127, 10000, replace=TRUE)
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
#'                 learningrate   =  1,
#'                 hidden_dim     = 16  )
#' }
#'     

trainr <- function(Y, X, model = NULL,learningrate, learningrate_decay = 1, momentum = 0, hidden_dim = c(10),network_type = "rnn",
                   numepochs = 1, sigmoid = c('logistic', 'Gompertz', 'tanh'), use_bias = F, batch_size = 1,
                   seq_to_seq_unsync=F,update_rule = "sgd",
                   epoch_function = c(epoch_print,epoch_annealing),
                   loss_function = loss_L1,...) {
  
  #  find sigmoid
  sigmoid <- match.arg(sigmoid)
  
  # coerce to array if matrix
  if(length(dim(X)) == 2){
    X <- array(X,dim=c(dim(X),1))
  }
  if(length(dim(Y)) == 2){
    Y <- array(Y,dim=c(dim(Y),1))
  }
  
  if(seq_to_seq_unsync){
    time_dim_input = dim(X)[2]
    store = array(0, dim = c(dim(X)[1],dim(X)[2]+dim(Y)[2]-1,dim(X)[3]))
    store[,1:dim(X)[2],] = X
    X = store
    store = array(0, dim = c(dim(X)[1],time_dim_input+dim(Y)[2]-1,dim(Y)[3]))
    store[,time_dim_input:dim(store)[2],] = Y
    Y = store
  }
  
  # check the consistency
  if(dim(X)[2] != dim(Y)[2] && !seq_to_seq_unsync){
    stop("The time dimension of X is different from the time dimension of Y. seq_to_seq_unsync is set to FALSE")
  }
  if(dim(X)[1] != dim(Y)[1]){
    stop("The sample dimension of X is different from the sample dimension of Y.")
  }
  
  
  # reverse the time dim if start from end
  # if(start_from_end){
  #   X <- X[,dim(X)[2]:1,,drop = F]
  #   Y <- Y[,dim(X)[2]:1,,drop = F]
  # }
  
  if(is.null(model)){
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
    model$batch_size              = batch_size
    model$learningrate            = learningrate
    model$learningrate_decay      = learningrate_decay ## this one should be in the ... arg and be here initially but he was supply before
    model$momentum                = momentum
    model$update_rule             = update_rule
    model$use_bias                = use_bias
    model$seq_to_seq_unsync       = seq_to_seq_unsync
    model$epoch_function          = epoch_function
    model$loss_function           = loss_function
    model$last_layer_error        = Y*0
    model$last_layer_delta        = Y*0
    
    if("epoch_model_function" %in% names(model)){
      stop("epoch_model_function is not used anymore, use epoch_function and return the model inside.")
    }
    
    if(seq_to_seq_unsync){ ## this will work for the training, we need something to make in work for the predictr
      model$time_dim_input = time_dim_input
    }
    
    if(model$update_rule == "adagrad"){
      message("adagrad update, loss function not used and momentum set to 0")
      model$momentum = 0
    }
    
    model <- init_r(model)
    
    # Storing errors, dim 1: samples, dim 2 is epochs, we could store also the time and variable dimension
    model$error <- array(0,dim = c(dim(Y)[1],model$numepochs))
  }else{
    message("retraining, all options except X, Y and the model itself are ignored, error are reseted")
    if(model$input_dim != dim(X)[3]){
      stop("input dim changed")
    }
    if(model$time_dim != dim(X)[2]){
      stop("time dim changed")
    }
    if(model$output_dim != dim(Y)[3]){
      stop("output dim changed")
    }
    if(seq_to_seq_unsync && model$time_dim_input != time_dim_input){ ## this will work for the training, we need something to make in work for the predictr
      stop("time input dim changed")
    }
    
    # Storing errors, dim 1: samples, dim 2 is epochs, we could store also the time and variable dimension
    model$error <- array(0,dim = c(dim(Y)[1],model$numepochs))
  }
  
  
  # training logic
  for(epoch in seq(model$numepochs)){
    model$current_epoch = epoch
    index = sample(seq(round(dim(Y)[1]/model$batch_size)),dim(Y)[1],replace = T)
    lj = list()
    for(i in seq(round(dim(Y)[1]/model$batch_size))){lj[[i]] = seq(dim(Y)[1])[index == i]}
    lj[unlist(lapply(lj,length)) <1] = NULL
    
    for (j in lj) {
      # generate input and output for the sample loop
      a = X[j,,,drop=F]
      c = Y[j,,,drop=F]
      
      # feed forward
      store = predictr(model,a,hidden = T,real_output = F)
      if(model$network_type == "rnn"){
        for(i in seq(length(model$synapse_dim) - 1)){
          model$store[[i]][j,,] = store[[i]]
        }
      }else if(model$network_type == "lstm" | model$network_type == "gru" ){
        for(i in seq(length(model$hidden_dim))){
          model$store[[i]][j,,,] = store[[i]]
        }
        model$store[[length(model$hidden_dim)+1]][j,,] = store[[length(model$hidden_dim)+1]] # output
      }
      
      
      # apply back propagation
      model = backprop_r(model,a,c,j)
      
      # apply the loss function, default is to apply L1 learning rate, vocabulary to verify.
      if(model$update_rule == "sgd"){
        model = model$loss_function(model)
      }
      
      # Applying the update
      model = update_r(model)
      
    } # end sample loop
    
    # epoch_function
    for(i in model$epoch_function){
      model <- i(model)
      if(!is.list(model)){stop("one epoch function didn't return the model.")}
    }
    
  } # end epoch loop
  
  # update best guess if error is minimal, will make more sens to store the weight...
  if(colMeans(model$error)[epoch] <= min(colMeans(model$error)[1:epoch])){
    model$store_best <- model$store
  }
  
  # clean model object, get rid of the update mainly, potentially other cleaning if not necessary in predictr
  # model = clean_r(model)
  
  attr(model, 'error') <- colMeans(model$error)
  
  # return output
  return(model)
  
}
