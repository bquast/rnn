## ----package-------------------------------------------------------------
library(rnn)

## ----code-rnn------------------------------------------------------------
rnn

## ----int2binary----------------------------------------------------------
int2binary(146, length=8)

## ----int2binary-code-----------------------------------------------------
int2binary

## ----sigmoid-------------------------------------------------------------
(a <- sigmoid(3))

## ----sigmoid-code--------------------------------------------------------
sigmoid

## ----sigmoid-der---------------------------------------------------------
sigmoid_output_to_derivative(a) # a was created above using sigmoid()

## ----sigmoid-der-code----------------------------------------------------
sigmoid_output_to_derivative

## ----help, eval=FALSE----------------------------------------------------
#  help('rnn')

## ----example-------------------------------------------------------------
# using the default of 10,000 iterations
rnn(binary_dim =  8,
    alpha      =  0.1,
    input_dim  =  2,
    hidden_dim = 10,
    output_dim =  1  )

