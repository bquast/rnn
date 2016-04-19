# replicable
set.seed(1)

# create training response numbers
X1 = sample(0:127, 7000, replace=TRUE)
X2 = sample(0:127, 7000, replace=TRUE)

# create training response numbers
Y <- X1 + X2

# convert to binary
X1 <- int2bin(X1)
X2 <- int2bin(X2)
Y  <- int2bin(Y)

# train the model
m1 <- trainr(Y,
             X1,
             X2,
             binary_dim =  8,
             alpha      =  0.1,
             input_dim  =  2,
             hidden_dim = 10,
             output_dim =  1,
             print      = 'full')

# create test inputs
A1 = int2bin( sample(0:127, 7000, replace=TRUE) )
A2 = int2bin( sample(0:127, 7000, replace=TRUE) )

# predict
B  <- predictr(m1,
               A1,
               A2,
               binary_dim =  8,
               alpha      =  0.1,
               input_dim  =  2,
               hidden_dim = 10,
               output_dim =  1,
               print      = 'full')

# inspect the differences              
expect_equal(sum(bin2int(B)), 927640)
