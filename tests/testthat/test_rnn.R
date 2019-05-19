# create training numbers
X1 = sample(0:127, 7000, replace=TRUE)
X2 = sample(0:127, 7000, replace=TRUE)

# create training response numbers
Y <- X1 + X2

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

# convert to binary
X1 <- int2bin(X1)
X2 <- int2bin(X2)
Y  <- int2bin(Y)

# Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
X <- array( c(X1,X2), dim=c(dim(X1),2) )
Y <- array( Y, dim=c(dim(Y),1) ) 

# train the model
model <- trainr(Y=Y[,dim(Y)[2]:1,,drop=F],
                X=X[,dim(X)[2]:1,,drop=F],
                learningrate   =  0.1,
                hidden_dim     =  c(10,10),
                numepochs      =  5,
                batch_size     = 100,
                momentum       =0,
                use_bias       = F,
                learningrate_decay = 1)

# create test inputs
A1 = int2bin( sample(0:127, 7000, replace=TRUE) )
A2 = int2bin( sample(0:127, 7000, replace=TRUE) )

# create 3d array: dim 1: samples; dim 2: time; dim 3: variables
A <- array( c(A1,A2), dim=c(dim(A1),2) )

# predict
B  <- predictr(model, A[,dim(A)[2]:1,,drop=F])[,dim(A)[2]:1]

# inspect the differences              
# expect_equal(sum(bin2int(B)), 888626)
# print(sum(bin2int(B)))
# print(sum(bin2int(A1))+sum(bin2int(A2)))
