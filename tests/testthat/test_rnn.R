# replicable
set.seed(1)

# create training inputs
X1 = sample(0:127, 7000, replace=TRUE)
X2 = sample(0:127, 7000, replace=TRUE)

# create training output
Y <- X1 + X2

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
