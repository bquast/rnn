#' Binary to Integer
#' 
#' @param binary input binary
#' @return integer representation
#' @export

bin2int <- function(binary){
  
  # round
  binary <- round(binary)
  
  # determine length of binary representation
  length <- dim(binary)[2]
  
  # apply to full matrix
  apply(binary, 1, b2i)
}

#' @describeIn bin2int individual Binary to Integer

b2i <- function(binary)
  packBits(as.raw(c(binary, rep(0, 32-length(binary) ))), 'integer')
