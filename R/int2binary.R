#' @name int2binary
#' @export
#' @title Integer 2 Binary
#' @description Converts an integer to binary form.


int2binary = function(x) {
  tail(rev(as.integer(intToBits(x))), binary_dim) }
