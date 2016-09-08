## shiny app for the rnn package

library(shiny)
source('binary.R')
source('finance.R')
source('cosinus.R')
source('oscillation.R')

ui <- navbarPage(title="rnn demo",
                 tabPanel("binary",
                          binaryUI('binary')
                          ),
                 tabPanel("cosinus",
                          cosinusUI('cosinus')
                 ),
                 tabPanel("finance",
                          financeUI('finance')
                 ),
                 tabPanel("oscillation",
                          oscillationUI('oscillation')
                 )
                 
)

server <- function(input,output,session){
  binary <- callModule(binaryServer,id='binary')
  cosinus <- callModule(cosinusServer,id='cosinus')
  finance <- callModule(financeServer,id='finance')
  oscillation <- callModule(oscillationServer,id='oscillation')
}

shinyApp(ui,server)
