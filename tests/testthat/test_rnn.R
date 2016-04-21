# replicable
set.seed(1)

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

# train the model
model <- trainr(Y=Y,
                X=X,
                learningrate   =  0.1,
                hidden_dim     = 10,
                numepochs      = 10,
                start_from_end = TRUE )

# create test inputs
A1 = int2bin( sample(0:127, 7000, replace=TRUE) )
A2 = int2bin( sample(0:127, 7000, replace=TRUE) )

# create 3d array: dim 1: samples; dim 2: time; dim 3: variables
A <- array( c(A1,A2), dim=c(dim(A1),2) )

# predict
B  <- predictr(model,
               A     )

# inspect the differences              
expect_equal(sum(bin2int(B)), 886614)
