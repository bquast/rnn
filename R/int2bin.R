#' Integer to Binary
#' 
#' @param integer input integer
#' @param length binary representation length
#' @return binary representation
#' @export

int2bin <- function(integer, length) {
  t(sapply(integer, i2b, length=length))
}


#' @export

i2b <- function(integer, length){
  rev(as.numeric(intToBits(integer))[1:length])
}

