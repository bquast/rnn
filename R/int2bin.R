#' Integer to Binary
#' 
#' @param integer input integer
#' @param output binary representation length
#' @return binary representation
#' @export

int2bin <- function(integer, length){
  rev(as.numeric(intToBits(integer))[1:length])
}
