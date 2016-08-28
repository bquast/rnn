## shiny app for the rnn package

library(shiny)
source('binary.R')
source('finance.R')
source('cosinus.R')


ui <- navbarPage(title="rnn demo",
                 tabPanel("binary",
                          binaryUI('binary')
                          ),
                 tabPanel("cosinus",
                          cosinusUI('cosinus')
                 ),
                 tabPanel("finance",
                          financeUI('finance')
                 )
                 
)

server <- function(input,output,session){
  binary <- callModule(binaryServer,id='binary')
  cosinus <- callModule(cosinusServer,id='cosinus')
  finance <- callModule(financeServer,id='finance')
}

shinyApp(ui,server)
