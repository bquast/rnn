# replicable
set.seed(1)

# create training response numbers
X1 = sample(0:127, 7000, replace=TRUE)
X2 = sample(0:127, 7000, replace=TRUE)

# create training response numbers
Y <- X1 + X2

# convert to binary
X1 <- int2bin(X1, length=8)
X2 <- int2bin(X2, length=8)
Y  <- int2bin(Y,  length=8)

# train the model
m1 <- trainr(Y,
             X1,
             X2,
             binary_dim =  8,
             alpha      =  0.1,
             input_dim  =  2,
             hidden_dim = 10,
             output_dim =  1   )

# create test inputs
A1 = sample(0:127, 7000, replace=TRUE)
A2 = sample(0:127, 7000, replace=TRUE)

# predict
B  <- predictr(m1,
               A1,
               A2,
               binary_dim =  8,
               alpha      =  0.1,
               input_dim  =  2,
               hidden_dim = 10,
               output_dim =  1   )

# inspect the differences              
expect_equal(sum(B), 927640)
