#' Integer to Binary
#' 
#' @param integer input integer
#' @param length binary representation length
#' @return binary representation
#' @export

int2bin <- function(integer, length=8) {
  t(sapply(integer, i2b, length=length))
}


#' @describeIn int2bin individual Integer to Binary

i2b <- function(integer, length=8){
  as.numeric(intToBits(integer))[1:length]
}

