# replicable
set.seed(123)

# create training inputs
X1 = sample(0:127, 7000, replace=TRUE)
X2 = sample(0:127, 7000, replace=TRUE)

# create training output
Y <- X1 + X2

# run the 
rnn(Y,
    X1,
    X2,
    binary_dim =  8,
    alpha      =  0.1,
    input_dim  =  2,
    hidden_dim = 10,
    output_dim =  1)
