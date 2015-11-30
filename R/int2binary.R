#' @name int2binary
#' @export
#' @importFrom utils tail
#' @title Integer 2 Binary
#' @description Converts an integer to binary form.
#' @param x integer
#' @param length length of binary representation


int2binary = function(x, length) {
  utils::tail(rev(as.integer(intToBits(x))), length) }
