## R script for oscillation LSTM test, not learning yet

set.seed(1)

X_dim_1 = 100000 # number of sampels
X_dim_2 = 30 # time dimension
X_dim_3 = 1 # number of variables, we will see later to make it more complex


X_event_proba = 0.1 # percent of 1 in the time series

X1 = array(sample(0:1,size=X_dim_1*X_dim_2,replace=TRUE,prob = c(1-X_event_proba,X_event_proba)),dim=c(X_dim_1,X_dim_2))
Y1 = array(rep(0,X_dim_1*X_dim_2),dim=c(X_dim_1,X_dim_2))


## rolling sum function
rollSum <- function(x){
  y = c()
  for(i in seq(length(x))){
    y = c(y,sum(x[1:i]) %% 2)
  }
  return(y)
}

for(i in seq(X_dim_1)){
  Y1[i,] <- rollSum(X1[i,])
}

# View(rbind(X[1,],Y[1,]))

set.seed(1)

# define some functions

## sigmoid function
sigmoid <- function(x, k=1, x0=0)
  1 / (1+exp( -k*(x-x0) ))

## derivative
sigmoid_output_to_derivative <- function(x)
  x*(1-x)

# input variables
alpha = 0.1
alpha_decay = 0.9999
momentum = 0.1
init_weight = 1
batch_size = 5
input_dim = 1
hidden_dim = 4
output_dim = 1
binary_dim = X_dim_2
output_size = 20



# initialise neural network weights
synapse_0_i = matrix(runif(n = input_dim *hidden_dim, min=-init_weight, max=init_weight), nrow=input_dim)
synapse_0_f = matrix(runif(n = input_dim *hidden_dim, min=-init_weight, max=init_weight), nrow=input_dim)
synapse_0_o = matrix(runif(n = input_dim *hidden_dim, min=-init_weight, max=init_weight), nrow=input_dim)
synapse_0_c = matrix(runif(n = input_dim *hidden_dim, min=-init_weight, max=init_weight), nrow=input_dim)
synapse_1   = matrix(runif(n = hidden_dim*output_dim, min=-init_weight, max=init_weight), nrow=hidden_dim)
synapse_h_i = matrix(runif(n = hidden_dim*hidden_dim, min=-init_weight, max=init_weight), nrow=hidden_dim)
synapse_h_f = matrix(runif(n = hidden_dim*hidden_dim, min=-init_weight, max=init_weight), nrow=hidden_dim)
synapse_h_o = matrix(runif(n = hidden_dim*hidden_dim, min=-init_weight, max=init_weight), nrow=hidden_dim)
synapse_h_c = matrix(runif(n = hidden_dim*hidden_dim, min=-init_weight, max=init_weight), nrow=hidden_dim)
synapse_b_1 = runif(n = output_dim, min=-init_weight, max=init_weight)
synapse_b_i = runif(n = hidden_dim, min=-init_weight, max=init_weight)
synapse_b_f = runif(n = hidden_dim, min=-init_weight, max=init_weight)
synapse_b_o = runif(n = hidden_dim, min=-init_weight, max=init_weight)
synapse_b_c = runif(n = hidden_dim, min=-init_weight, max=init_weight)

# initialise state cell
c_t_m1      = matrix(0, nrow=1, ncol = hidden_dim)

# initialise synapse updates
synapse_0_i_update = matrix(0, nrow = input_dim, ncol = hidden_dim)
synapse_0_f_update = matrix(0, nrow = input_dim, ncol = hidden_dim)
synapse_0_o_update = matrix(0, nrow = input_dim, ncol = hidden_dim)
synapse_0_c_update = matrix(0, nrow = input_dim, ncol = hidden_dim)
synapse_1_update   = matrix(0, nrow = hidden_dim, ncol = output_dim)
synapse_h_i_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)
synapse_h_f_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)
synapse_h_o_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)
synapse_h_c_update = matrix(0, nrow = hidden_dim, ncol = hidden_dim)
synapse_b_1_update = rep(0, output_dim)
synapse_b_i_update = rep(0, hidden_dim)
synapse_b_f_update = rep(0, hidden_dim)
synapse_b_o_update = rep(0, hidden_dim)
synapse_b_c_update = rep(0, hidden_dim)

# training logic
for (j in 1:length(X1)) {
  # select input variables
  a = X1[j,]
  
  # response variable
  c = Y1[j,]
  
  # where we'll store our best guesss (binary encoded)
  d = matrix(0, nrow = 1, ncol = binary_dim)
  
  overallError = 0
  
  layer_2_deltas = matrix(0)
  layer_1_values = matrix(0, nrow=1, ncol = hidden_dim)
  # initialise state cell
  c_t_m1      = matrix(0, nrow=1, ncol = hidden_dim)
  
  # moving along the positions in the binary encoding
  for (position in 1:binary_dim) {
    
    # generate input and output
    X = a[position]
    y = c[position]
    
    # hidden layer (input ~+ prev_hidden)
    i_t     = sigmoid((X%*%synapse_0_i) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h_i) + synapse_b_i) # add bias?
    f_t     = sigmoid((X%*%synapse_0_f) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h_f) + synapse_b_f) # add bias?
    o_t     = sigmoid((X%*%synapse_0_o) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h_o) + synapse_b_o) # add bias?
    c_in_t  = tanh(   (X%*%synapse_0_c) + (layer_1_values[dim(layer_1_values)[1],] %*% synapse_h_c) + synapse_b_c)
    c_t     = (f_t * c_t_m1[dim(layer_1_values)[1],]) + (i_t * c_in_t)
    layer_1 = o_t * tanh(c_t)
    c_t_m1  = rbind(c_t_m1, c_t)
    
    # output layer (new binary representation)
    layer_2 = sigmoid(layer_1 %*% synapse_1 + synapse_b_1)
    
    # did we miss?... if so, by how much?
    layer_2_error = y - layer_2
    layer_2_deltas = rbind(layer_2_deltas, layer_2_error * sigmoid_output_to_derivative(layer_2))
    overallError = overallError + round(abs(layer_2_error))
    
    # decode estimate so we can print it out
    d[position] = round(layer_2)
    
    # store hidden layer so we can print it out
    layer_1_values = rbind(layer_1_values, layer_1)
    
  }
  
  future_layer_1_i_delta = matrix(0, nrow = 1, ncol = hidden_dim)
  future_layer_1_f_delta = matrix(0, nrow = 1, ncol = hidden_dim)
  future_layer_1_o_delta = matrix(0, nrow = 1, ncol = hidden_dim)
  future_layer_1_c_delta = matrix(0, nrow = 1, ncol = hidden_dim)
  
  for (position in 1:binary_dim) {
    
    X = a[binary_dim-(position-1)]
    layer_1 = layer_1_values[dim(layer_1_values)[1]-(position-1),]
    prev_layer_1 = layer_1_values[dim(layer_1_values)[1]-position,]
    
    # error at output layer
    layer_2_delta = layer_2_deltas[dim(layer_2_deltas)[1]-(position-1),]
    # error at hidden layer
    layer_1_i_delta = (future_layer_1_i_delta %*% t(synapse_h_i) + layer_2_delta %*% t(synapse_1)) *
      sigmoid_output_to_derivative(layer_1)
    layer_1_f_delta = (future_layer_1_f_delta %*% t(synapse_h_f) + layer_2_delta %*% t(synapse_1)) *
      sigmoid_output_to_derivative(layer_1)
    layer_1_o_delta = (future_layer_1_o_delta %*% t(synapse_h_o) + layer_2_delta %*% t(synapse_1)) *
      sigmoid_output_to_derivative(layer_1)
    layer_1_c_delta = (future_layer_1_c_delta %*% t(synapse_h_c) + layer_2_delta %*% t(synapse_1)) *
      sigmoid_output_to_derivative(layer_1)
    
    
    # let's update all our weights so we can try again
    synapse_1_update   = synapse_1_update   + matrix(layer_1)      %*% layer_2_delta
    synapse_h_i_update = synapse_h_i_update + matrix(prev_layer_1) %*% layer_1_i_delta
    synapse_h_f_update = synapse_h_f_update + matrix(prev_layer_1) %*% layer_1_f_delta
    synapse_h_o_update = synapse_h_o_update + matrix(prev_layer_1) %*% layer_1_o_delta
    synapse_h_c_update = synapse_h_c_update + matrix(prev_layer_1) %*% layer_1_c_delta
    synapse_0_i_update = synapse_0_i_update + t(X) %*% layer_1_i_delta
    synapse_0_f_update = synapse_0_f_update + t(X) %*% layer_1_f_delta
    synapse_0_o_update = synapse_0_o_update + t(X) %*% layer_1_o_delta
    synapse_0_c_update = synapse_0_c_update + t(X) %*% layer_1_c_delta
    synapse_b_1_update = synapse_b_1_update + layer_2_delta
    synapse_b_i_update = synapse_b_i_update + layer_1_i_delta
    synapse_b_f_update = synapse_b_f_update + layer_1_f_delta
    synapse_b_o_update = synapse_b_o_update + layer_1_o_delta
    synapse_b_c_update = synapse_b_c_update + layer_1_c_delta
    
    future_layer_1_i_delta = layer_1_i_delta
    future_layer_1_f_delta = layer_1_f_delta
    future_layer_1_o_delta = layer_1_o_delta
    future_layer_1_c_delta = layer_1_c_delta
  }
  if(j %% batch_size ==0) {
    synapse_0_i = synapse_0_i + ( synapse_0_i_update * alpha )
    synapse_0_f = synapse_0_f + ( synapse_0_f_update * alpha )
    synapse_0_o = synapse_0_o + ( synapse_0_o_update * alpha )
    synapse_0_c = synapse_0_c + ( synapse_0_c_update * alpha )
    synapse_1   = synapse_1   + ( synapse_1_update   * alpha )
    synapse_h_i = synapse_h_i + ( synapse_h_i_update * alpha )
    synapse_h_f = synapse_h_f + ( synapse_h_f_update * alpha )
    synapse_h_o = synapse_h_o + ( synapse_h_o_update * alpha )
    synapse_h_c = synapse_h_c + ( synapse_h_c_update * alpha )
    synapse_b_1   = synapse_b_1   + ( synapse_b_1_update   * alpha )
    synapse_b_i = synapse_b_i + ( synapse_b_i_update * alpha )
    synapse_b_f = synapse_b_f + ( synapse_b_f_update * alpha )
    synapse_b_o = synapse_b_o + ( synapse_b_o_update * alpha )
    synapse_b_c = synapse_b_c + ( synapse_b_c_update * alpha )
    
    alpha = alpha * alpha_decay
    
    synapse_0_i_update = synapse_0_i_update * momentum
    synapse_0_f_update = synapse_0_f_update * momentum
    synapse_0_o_update = synapse_0_o_update * momentum
    synapse_0_c_update = synapse_0_c_update * momentum
    synapse_1_update   = synapse_1_update   * momentum
    synapse_h_i_update = synapse_h_i_update * momentum
    synapse_h_f_update = synapse_h_f_update * momentum
    synapse_h_o_update = synapse_h_o_update * momentum
    synapse_h_c_update = synapse_h_c_update * momentum
    synapse_b_1_update = synapse_b_1_update * momentum
    synapse_b_i_update = synapse_b_i_update * momentum
    synapse_b_f_update = synapse_b_f_update * momentum
    synapse_b_o_update = synapse_b_o_update * momentum
    synapse_b_c_update = synapse_b_c_update * momentum
  }
  
  
  
  # print out progress
  if(j %% output_size ==0) {
    print(paste("Error:", overallError," - alpha:",alpha))
    print(paste("Input:", paste(a, collapse = "")))
    print(paste("True: ", paste(c, collapse = "")))
    print(paste("Pred: ", paste(d, collapse = "")))
    out = 0
    for (x in 1:length(d)) {
      out[x] = rev(d)[x]*2^(x-1) }
    print("----------------")
  }
}