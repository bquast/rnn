#' @name sigmoid
#' @export
#' @title Sigmoid
#' @description computes sigmoid nonlinearity
#' @param  x number
#' 

sigmoid <- function(x, method=c('logistic', 'Gompertz', 'ogive'), inverse=FALSE, ...) {
  #  find method
  method <- match.arg(method)
  
  if (method=='logistic' && inverse==FALSE) {
    return( logistic(x, ...) )
  } else if (method=='Gompertz' && inverse==FALSE) {
    return( Gompertz(x, ...) )
  } else if (method=='logistic' && inverse==TRUE) {
    return ( logit(x) )
  } else if (method=='Gompertz' && inverse==TRUE) {
    return( inverse_Gompertz(x) )
  }
  
}

logistic <- function(x, k=1, x0=0)
  1 / (1+exp( -k*(x-x0) ))
  
Gompertz <- function(x, a=1, b=1, c=1)
  a*exp(-b*exp(-c*x))

logit <- function(x)
  log( x / (1-x) )

inverse_Gompertz <- function(x)
  -1*log(-1*log(x))
  

#' @name sigmoid_output_to_derivative
#' @export
#' @title Sigmoid Derivative
#' @description Convert output of sigmoid function to its derivative.
#' @param output sigmoid value

sigmoid_output_to_derivative <- function(x)
  x*(1-x)
