#' @name rnn
#' @docType package
#' @title recurrent neural network
#' @author
#' Bastiaan Quast \email{bquast@@gmail.com}
NULL
#' @name tlc_data
#' @docType data
#' @title tlc_data Example
#' @description the propolis dataset
NULL
.onAttach <- function(...) {
  packageStartupMessage('If you use rnn for data analysis,
please cite both R and rnn,
using citation() and citation("rnn") respectively.
')}
