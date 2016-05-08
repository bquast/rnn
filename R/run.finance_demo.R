##' @name run.finance_demo
##' @export
##' @importFrom shiny runApp
##' @title Demo app on finance data
##' @description Function to launch the finance_demo app
##' @param port if not NULL will deploy on the local network on this port, just look for the IP of the host and go for example on 192.168.1.20:port to access the app from anywhere in the network
##' @author Dimitri Fichou
##' @examples
##' \dontrun{
##' run.finance_demo()
##' }

run.finance_demo <- function(port=NULL) {
  appDir <- system.file("shinyapps", "finance_demo", package = "rnn")
  if (appDir == "") {
    stop("Could not find example directory. Try re-installing `mypackage`.", call. = FALSE)
  }
  if(is.null(port)){
    shiny::runApp(appDir, display.mode = "normal",launch.browser = T)
  }else{
    shiny::runApp(appDir, display.mode = "normal",
                  launch.browser = T,host = '0.0.0.0',port=port)
  }
}
