# binary addition demo for rnn package
# just an app.R file to keep it simple, hopefully, it will stay this way

library(rnn)
library(shiny)
library(sigmoid)
load("www/finance.Rdata")

financeUI <- function(id){
  ns = NS(id)
  sidebarLayout(
    sidebarPanel(
      h4("Data parameters"),
      selectizeInput(ns("target"),"currency to predict",choices=c("EURUSD"=1,"CHFUSD"=2,"GBPUSD"=3,"JPYUSD"=4),selected=1),
      numericInput(ns("prediction_gap"),"Number of day in advance to predict, the bigger, the more difficult",1),
      numericInput(ns("training_amount"),"Number of days of training, the bigger, the easier",1000),
      h4("Network parameters"),
      numericInput(ns("epoch"),"number of epochs",50),
      numericInput(ns("hidden"),"number of hidden units",10),
      numericInput(ns("lr"),"learning rate",0.01),
      actionButton(ns("go"),"Train")
    ),
    mainPanel(plotOutput(ns("PredictPlot"),height = "800px"))
  )
}

financeServer <- function(input, output,session) {
  X.train <- reactive({
    # print(length(1:input$training_amount))
    X <- X[,1:input$training_amount,]
    array(X,dim=c(1,input$training_amount,4))
  })
  y.train <- reactive({
    # print(length(1:input$training_amount+input$prediction_gap))
    y <- X[,1:input$training_amount+input$prediction_gap,as.numeric(input$target)]
    matrix(y, ncol=input$training_amount)
  })
  model <- eventReactive(input$go,{
    withProgress(message = "Training network", value=0, {
      model <- trainr(X = X.train(),
                      Y = y.train(),
                      learningrate = input$lr,
                      numepochs = input$epoch,
                      hidden_dim = input$hidden,
                      use_bias = T
                      )
      return(model)
    })
  })
  X.test <- reactive({
    # print(length(1:(max(dim(X))-input$prediction_gap)))
    X[,1:(dim(X)[2]-input$prediction_gap),,drop=F]
  })
  y.test <- reactive({
    # print(length((1+input$prediction_gap):max(dim(X))))
    y <- X[,(1+input$prediction_gap):dim(X)[2],as.numeric(input$target)]
    matrix(y, ncol=(max(dim(X))-input$prediction_gap))
  })
  y.pred <- reactive({
    predictr(model(), X.test())
  })
  output$PredictPlot <- renderPlot({
    par(mfrow=c(2,1))
    for(i in seq(4)){
      plot(X.test()[,,i],yaxt="n",type="l",col=i,ylab="")
      par(new=T)
    }
    title(main="Independent variables")
    par(new=F)
    # y.test()[1,] %>% length %>% print
    plot(y.test()[1,],yaxt="n",type="l",col="green",main="Dependent variables",ylab="")
    par(new=T)
    # y.pred()[1,] %>% length %>% print
    plot(y.pred()[1,],yaxt="n",type="l",col="red",ylab="")
    abline(v=input$training_amount)
    legend("topright",col = c("green","red"),pch=1,legend = c("target","prediction"))
  })
}
