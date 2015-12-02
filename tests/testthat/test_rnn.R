# create training inputs
x1 = sample(0:127, 7000, replace=TRUE)
x2 = sample(0:127, 7000, replace=TRUE)

# create training output
y <- x1 + x2

# run the 
rnn(y,
    x1,
    x2,
    binary_dim =  8,
    alpha      =  0.1,
    input_dim  =  2,
    hidden_dim = 10,
    output_dim =  1)
