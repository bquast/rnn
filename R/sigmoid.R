#' @name sigmoid
#' @export
#' @title Sigmoid
#' @description computes sigmoid nonlinearity
#' @param  x number
#' 

sigmoid = function(x) {
  output = 1 / (1+exp(-x))
  return(output)            }

#' @name sigmoid_output_to_derivative
#' @export
#' @title Sigmoid Derivative
#' @description Convert output of sigmoid function to its derivative.
#' @param output sigmoid value

sigmoid_output_to_derivative = function(output) {
  return( output*(1-output) )                      }
