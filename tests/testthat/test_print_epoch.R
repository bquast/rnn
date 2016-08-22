# replicable
set.seed(1)
# function int2bin and bin2int
int2bin <- function(integer, length=8) {
  t(sapply(integer, i2b, length=length))
}

i2b <- function(integer, length=8){
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


# create training numbers
X1 = sample(0:127, 7000, replace=TRUE)
X2 = sample(0:127, 7000, replace=TRUE)

# create training response numbers
Y <- X1 + X2

# convert to binary
X1 <- int2bin(X1)
X2 <- int2bin(X2)
Y  <- int2bin(Y)

# Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
X <- array( c(X1,X2), dim=c(dim(X1),2) )
Y <- array( Y, dim=c(dim(Y),1) ) 

# create test numbers
X1_test = sample(0:127, 700, replace=TRUE)
X2_test = sample(0:127, 700, replace=TRUE)

# create test response numbers
Y_test <- X1_test + X2_test

# convert to binary
X1_test <- int2bin(X1_test)
X2_test <- int2bin(X2_test)
Y_test  <- int2bin(Y_test)

# Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
X_test <- array( c(X1_test,X2_test), dim=c(dim(X1_test),2) )
# Y_test <- array( Y_test, dim=c(dim(Y_test),1) ) 

plot_error = function(model){
  error = colMeans(model$error)[1:model$current_epoch]
  par(mfrow=c(2,1))
  plot(error,type="l",xlim=c(0,model$numepochs),ylim=c(0,max(error)),xlab="epoch",ylab = "error",main = paste0("training set, epoch ",model$current_epoch))
  error = mean(apply(model$Y_test - predictr(model,model$X_test),1,function(x){sum(abs(x))}))
  if(is.null(model$error_test)){model$error_test = c()}
  model$error_test <- c(model$error_test,error)
  plot(model$error_test,type="l",ylim=c(0,max(model$error_test)),xlim=c(0,model$numepochs),xlab="epoch",ylab = "error",main = paste0("test set, epoch ",model$current_epoch))
  return(model)
  }

# train the model
model <- trainr(Y=Y[,dim(Y)[2]:1,,drop=F],
                X=X[,dim(X)[2]:1,,drop=F],
                learningrate   =  0.1,
                hidden_dim     =  c(10,10),
                numepochs      =  20,
                batch_size     = 100,
                momentum       =0,
                use_bias       = F,
                learningrate_decay = 1,
                epoch_model_function = c(epoch_annealing,plot_error),
                epoch_function = c(epoch_print),
                X_test = X_test[,dim(X_test)[2]:1,,drop=F],
                Y_test=Y_test[,dim(Y_test)[2]:1,drop=F]
                )

set.seed(2) # need a new seed as RNG as moved during trainr because of bias generation, in order to compare before after the bias implementation

# create test inputs
A1 = int2bin( sample(0:127, 7000, replace=TRUE) )
A2 = int2bin( sample(0:127, 7000, replace=TRUE) )

# create 3d array: dim 1: samples; dim 2: time; dim 3: variables
A <- array( c(A1,A2), dim=c(dim(A1),2) )

# predict
B  <- predictr(model, A[,dim(A)[2]:1,,drop=F])[,dim(A)[2]:1]

# inspect the differences              
# expect_equal(sum(bin2int(B)), 890137)
print(sum(bin2int(B)))
# print(sum(bin2int(A1))+sum(bin2int(A2)))


