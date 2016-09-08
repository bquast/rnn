# binary addition demo for rnn package
# just an app.R file to keep it simple, hopefully, it will stay this way

library(rnn)
library(shiny)
library(sigmoid)

cosinusUI <- function(id){
  ns = NS(id)
  sidebarLayout(
    sidebarPanel(width = 3,
                 h3("data"),
                 numericInput(ns("time_dim"),"time dimension",200),
                 numericInput(ns("sample_dim_train"),"training sample dimension",9),
                 numericInput(ns("sample_dim_test"),"testing sample dimension",10),
                 h3("network"),
                 numericInput(ns("layers"),"number of hidden layers",1),
                 uiOutput(ns("hidden")),
                 numericInput(ns("learningrate"),"learningrate",0.01),
                 numericInput(ns("batchsize"),"batchsize",1),
                 numericInput(ns("numepochs"),"numepochs",200),
                 numericInput(ns("momentum"),"momentum",0),
                 checkboxInput(ns("use_bias"),"use_bias",F),
                 numericInput(ns("learningrate_decay"),"learningrate_decay",1),
                 actionButton(ns("go"),"Train")
                 
    ),
    mainPanel(
      # img(src="binary.gif"),
      plotOutput(ns("error")),
      numericInput(ns("test_sample"),"test sample index to look at",1),
      numericInput(ns("test_epoch"),"epoch to look at",1),
      plotOutput(ns("plot_observe"),height = "800px"),
      tableOutput(ns("table_observe"))
    )
  )
}

cosinusServer <- function(input, output,session) {
  output$hidden <- renderUI({
    ns <- session$ns
    x <- list()
    for(i in seq(input$layers)){
      x[[i]] <- numericInput(inputId = ns(paste0('hidden',i)),label = paste0('Number of unit in the layer number ',i),value = 16)
    }
    return(tagList(x))
  })
  
  train_set <- reactive({
    # replicable
    set.seed(1)
    time_dim = input$time_dim
    sample_dim <- input$sample_dim_train
    X <- data.frame()
    Y <- data.frame()
    
    # Genereate a bias in phase
    bias_phase <- rnorm(sample_dim)
    # Generate a bias in frequency
    bias_frequency = runif(sample_dim,min=5,max=25)
    
    # Generate the noisy time series, cosinus for X and sinus for Y, with a random bias in phase and in frequency
    for(i in seq(sample_dim)){
      X <- rbind(X,sin(seq(time_dim)/bias_frequency[i]+bias_phase[i])+rnorm(time_dim,mean=0,sd=0.1))
      Y <- rbind(Y,cos(seq(time_dim)/bias_frequency[i]+bias_phase[i])+rnorm(time_dim,mean=0,sd=0.1))
    }
    X <- as.matrix(X)
    Y <- as.matrix(Y)
    
    # Normalize between 0 and 1 for the sigmoid
    X <- (X-min(X))/(max(X)-min(X))
    Y <- (Y-min(Y))/(max(Y)-min(Y))
    
    list(X,Y)
  })
  
  test_set <- reactive({
    # replicable
    set.seed(2)
    time_dim = input$time_dim
    sample_dim <- input$sample_dim_test
    X <- data.frame()
    Y <- data.frame()
    
    # Genereate a bias in phase
    bias_phase <- rnorm(sample_dim)
    # Generate a bias in frequency
    bias_frequency = runif(sample_dim,min=5,max=25)
    
    # Generate the noisy time series, cosinus for X and sinus for Y, with a random bias in phase and in frequency
    for(i in seq(sample_dim)){
      X <- rbind(X,sin(seq(time_dim)/bias_frequency[i]+bias_phase[i])+rnorm(time_dim,mean=0,sd=0.1))
      Y <- rbind(Y,cos(seq(time_dim)/bias_frequency[i]+bias_phase[i])+rnorm(time_dim,mean=0,sd=0.1))
    }
    X <- as.matrix(X)
    Y <- as.matrix(Y)
    
    # Normalize between 0 and 1 for the sigmoid
    X <- (X-min(X))/(max(X)-min(X))
    Y <- (Y-min(Y))/(max(Y)-min(Y))
    
    list(X,Y)
  })
  
  model <- eventReactive(input$go,{
    print_test = function(model){
      message(paste0("Trained epoch: ",model$current_epoch," - Learning rate: ",model$learningrate))
      message(paste0("Epoch error: ",colMeans(model$error)[model$current_epoch]))
      pred = predictr(model,model$X_test,real_output = T,hidden = T)
      if(is.null(model$test_error)){model$test_error = array(0,dim = c(dim(model$Y_test)[1],model$numepochs))}
      model$test_error[,model$current_epoch] <- apply(model$Y_test - pred[[length(pred)]],1,function(x){sum(abs(x))})
      if(is.null(model$test_store)){model$test_store = list()}
      model$test_store[[length(model$test_store)+1]] <- pred
      return(model)
    }
    hidden_dim = c()
    for(i in seq(input$layers)){
      hidden_dim[i] <- input[[paste0('hidden',i)]]
    }
    withProgress(message = "Training network", value=0, {
      model <- trainr(X = train_set()[[1]],
                      Y = train_set()[[2]],
                      X_test = test_set()[[1]],
                      Y_test = test_set()[[2]],
                      learningrate = input$learningrate,
                      batch_size = input$batchsize,
                      numepochs = input$numepochs,
                      momentum = input$momentum,
                      use_bias = input$use_bias,
                      learningrate_decay = input$learningrate_decay,
                      hidden_dim = hidden_dim,
                      epoch_function = c(epoch_annealing,print_test)
      )
      return(model)
    })
  })
  
  
  output$error <- renderPlot({
    error = colMeans(model()$error)
    par(mfrow=c(1,2))
    plot(error,type="l",xlab="epoch",ylab = "error",main = "training set")
    error = colMeans(model()$test_error)
    plot(error,type="l",xlab="epoch",ylab = "error",main = "test set")      
  })
  
  output$table_observe <- renderTable({
    hidden_dim = c()
    for(i in seq(input$layers)){
      hidden_dim[i] <- input[[paste0('hidden',i)]]
    }
    data = matrix(0,ncol=input$time_dim,nrow=3)
    data[1,] = test_set()[[1]][input$test_sample,]
    data[2,]   = test_set()[[2]][input$test_sample,]
    data[3,]   = model()$test_store[[input$test_epoch]][[length(hidden_dim)+1]][input$test_sample,]
    rownames(data) <- c("X","Y","pred")
    for(i in seq(length(hidden_dim))){
      truc = array(model()$test_store[[input$test_epoch]][[i]][input$test_sample,,],dim=c(hidden_dim[i],input$time_dim))
      rownames(truc) = paste0("layer ",i," - hidden ",seq(hidden_dim[i]))
      data = rbind(data,truc)
    }
    
    data
  },digits = 0)
  
  output$plot_observe <- renderPlot({
    hidden_dim = c()
    for(i in seq(input$layers)){
      hidden_dim[i] <- input[[paste0('hidden',i)]]
    }
    
    data = matrix(0,ncol=input$time_dim,nrow=3)
    data[1,] = test_set()[[1]][input$test_sample,]
    data[2,]   = test_set()[[2]][input$test_sample,]
    data[3,]   = model()$test_store[[input$test_epoch]][[length(hidden_dim)+1]][input$test_sample,]
    rownames(data) <- c("X","Y","pred")
    
    layout(matrix(seq((length(hidden_dim)+1)*2),nrow=length(hidden_dim)+1,byrow = T), width=c(4,1)) 
    
    par(mar=c(5,4,4,0)) #No margin on the right side
    matplot(t(data),type="l",main = "Input and output layers",ylab = "units states",xlab = "time dim")
    par(mar=c(5,0,4,2)) #No margin on the left side
    plot(c(0,1),type="n", axes=F, xlab="", ylab="")
    legend("center", rownames(data),col=seq_len(4),cex=0.8,fill=seq_len(4))
    
    for(i in seq(length(hidden_dim))){
      data = array(model()$test_store[[input$test_epoch]][[i]][input$test_sample,,],dim=c(hidden_dim[i],input$time_dim))
      rownames(data) = paste0("layer ",i," - hidden ",seq(hidden_dim[i]))
      par(mar=c(5,4,4,0)) #No margin on the right side
      matplot(t(data),type="l",main = paste0("hidden layer ",i),ylab = "units states",xlab = "time dim")
      par(mar=c(5,0,4,2)) #No margin on the left side
      plot(c(0,1),type="n", axes=F, xlab="", ylab="")
      legend("center", rownames(data),col=seq_len(4),cex=0.8,fill=seq_len(4))
    }
  })
}
