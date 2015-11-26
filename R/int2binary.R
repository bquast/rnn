#' @name int2binary
#' @export
#' @title Integer 2 Binary
#' @description Converts an integer to binary form.
#' @param x integer
#' @param length length of binary representation


int2binary = function(x, length) {
  tail(rev(as.integer(intToBits(x))), length) }
