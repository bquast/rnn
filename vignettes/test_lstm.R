# function int2bin and bin2int
int2bin <- function(integer, length=time_dim) {
  t(sapply(integer, i2b, length=length))
}

i2b <- function(integer, length=time_dim){
  rev(as.numeric(intToBits(integer))[1:length])
}

bin2int <- function(binary){
  # round
  binary <- round(binary)
  # determine length of binary representation
  length <- dim(binary)[2]
  # apply to full matrix
  apply(binary, 1, b2i)
}

b2i <- function(binary)
  packBits(as.raw(rev(c(rep(0, 32-length(binary) ), binary))), 'integer')

time_dim = 8

# create training numbers
X1 = sample(0:(2^(time_dim-1)-1), 7000, replace=TRUE)
X2 = sample(0:(2^(time_dim-1)-1), 7000, replace=TRUE)

# create training response numbers
Y <- X1 + X2

# convert to binary
X1 <- int2bin(X1)
X2 <- int2bin(X2)
Y  <- int2bin(Y)

# Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
X <- array( c(X1,X2), dim=c(dim(X1),2) )
Y <- array( Y, dim=c(dim(Y),1) ) 

X = X[,dim(X)[2]:1,,drop=F]
Y = Y[,dim(Y)[2]:1,,drop=F]

# test set
set.seed(2)

# create training numbers
X1_test = sample(0:(2^(time_dim-1)-1), 7000, replace=TRUE)
X2_test = sample(0:(2^(time_dim-1)-1), 7000, replace=TRUE)

# create training response numbers
Y_test <- X1_test + X2_test

# convert to binary
X1_test <- int2bin(X1_test)
X2_test <- int2bin(X2_test)
Y_test  <- int2bin(Y_test)

# Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
X_test <- array( c(X1_test,X2_test), dim=c(dim(X1_test),2) )
Y_test <- array( Y_test, dim=c(dim(Y_test),1) ) 

sum(bin2int(Y_test))
sum(bin2int(X_test[,,1]))+sum(bin2int(X_test[,,2]))

X_test = X_test[,dim(X_test)[2]:1,,drop=F]
Y_test = Y_test[,dim(Y_test)[2]:1,,drop=F]

print_test = function(model){
  message(paste0("Trained epoch: ",model$current_epoch," - Learning rate: ",model$learningrate))
  message(paste0("Epoch error: ",colMeans(model$error)[model$current_epoch]))
  pred = predictr(model,model$X_test)
  message(paste0("Test set: target/predict - ",model$target_test,"/",sum(bin2int(pred))))
  print(paste("perfect:",sum(apply(Y_test[,,1] - round(pred[,]),1,sum) == 0),"/",nrow(Y_test)))
  n = sample(seq(nrow(pred)),1)
  print(paste("Pred:", paste(round(pred[n,]), collapse = " ")))
  print(paste("True:", paste(model$Y_test[n,,], collapse = " ")))
  print("- - - - - - - - - - -")
  n = sample(seq(nrow(pred)),1)
  print(paste("Pred:", paste(round(pred[n,]), collapse = " ")))
  print(paste("True:", paste(model$Y_test[n,,], collapse = " ")))
}

# train the model
model <- trainr(Y=Y,
                X=X,
                X_test = X_test,
                target_test = sum(bin2int(Y_test)),
                Y_test = Y_test,
                learningrate   =  0.01,
                hidden_dim     =  c(16),
                batch_size     = 100,
                numepochs      =  50,
                momentum       =0,
                use_bias       = T,
                network_type = "lstm",
                # sigmoid = "Gompertz",
                clipping = 1000000,
                learningrate_decay = 0.95,
                epoch_function = c(print_test)
                )