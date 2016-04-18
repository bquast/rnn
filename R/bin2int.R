#' Binary to Integer
#' 
#' @param binary input binary
#' @return binary representation
#' @export

bin2int <- function(binary){
  length <- length(binary)
  packBits(as.raw(rev(c(rep(0, 32-length), binary))), 'integer')
}
