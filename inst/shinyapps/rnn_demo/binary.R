# binary addition demo for rnn package
# just an app.R file to keep it simple, hopefully, it will stay this way

library(rnn)
library(shiny)
library(sigmoid)

# function int2bin and bin2int
int2bin <- function(integer, length=time_dim) {
  t(sapply(integer, i2b, length=length))
}

i2b <- function(integer, length=time_dim){
  rev(as.numeric(intToBits(integer))[1:length])
}

bin2int <- function(binary){
  # round
  binary <- round(binary)
  # determine length of binary representation
  length <- dim(binary)[2]
  # apply to full matrix
  apply(binary, 1, b2i)
}

b2i <- function(binary){
  packBits(as.raw(rev(c(rep(0, 32-length(binary) ), binary))), 'integer')
}



binaryUI <- function(id){
  ns = NS(id)
  sidebarLayout(
    sidebarPanel(width = 3,
      h3("data"),
      numericInput(ns("time_dim"),"time dimension",8),
      numericInput(ns("sample_dim_train"),"training sample dimension",7000),
      numericInput(ns("sample_dim_test"),"testing sample dimension",7000),
      h3("network"),
      numericInput(ns("layers"),"number of hidden layers",2),
      uiOutput(ns("hidden")),
      numericInput(ns("learningrate"),"learningrate",0.1),
      numericInput(ns("batchsize"),"batchsize",100),
      numericInput(ns("numepochs"),"numepochs",5),
      numericInput(ns("momentum"),"momentum",0.5),
      checkboxInput(ns("use_bias"),"use_bias",T),
      numericInput(ns("learningrate_decay"),"learningrate_decay",1),
      actionButton(ns("go"),"Train")
      
    ),
    mainPanel(
      # img(src="binary.gif"),
      plotOutput(ns("error")),
      numericInput(ns("test_sample"),"test sample index to look at",1),
      numericInput(ns("test_epoch"),"epoch to look at",1),
      column(6,
          tableOutput(ns("table_observe"))
          ),
      column(6,
          plotOutput(ns("plot_observe"),height = "800px")
          )
      
    )
  )
}

binaryServer <- function(input, output,session) {
  output$hidden <- renderUI({
    ns <- session$ns
    x <- list()
    for(i in seq(input$layers)){
      x[[i]] <- numericInput(inputId = ns(paste0('hidden',i)),label = paste0('Number of unit in the layer number ',i),value = 10)
    }
    return(tagList(x))
  })
  
  train_set <- reactive({
    time_dim = input$time_dim
    
    # create training numbers
    X1 = sample(0:(2^(input$time_dim-1)-1), input$sample_dim_train, replace=TRUE)
    X2 = sample(0:(2^(input$time_dim-1)-1), input$sample_dim_train, replace=TRUE)
    
    # create training response numbers
    Y <- X1 + X2
    
    # convert to binary
    X1 <- int2bin(X1,input$time_dim)
    X2 <- int2bin(X2,input$time_dim)
    Y  <- int2bin(Y,input$time_dim)
    
    # Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
    X <- array( c(X1,X2), dim=c(dim(X1),2) )
    Y <- array( Y, dim=c(dim(Y),1) ) 
    
    X = X[,dim(X)[2]:1,,drop=F]
    Y = Y[,dim(Y)[2]:1,,drop=F]
    
    list(X = X, Y = Y)
  })
  
  test_set <- reactive({
    # replicable
    set.seed(2)
    time_dim = input$time_dim
    # create training numbers
    X1 = sample(0:(2^(input$time_dim-1)-1), input$sample_dim_test, replace=TRUE)
    X2 = sample(0:(2^(input$time_dim-1)-1), input$sample_dim_test, replace=TRUE)
    
    # create training response numbers
    Y <- X1 + X2
    
    # convert to binary
    X1 <- int2bin(X1,input$time_dim)
    X2 <- int2bin(X2,input$time_dim)
    Y  <- int2bin(Y,input$time_dim)
    
    # Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
    X <- array( c(X1,X2), dim=c(dim(X1),2) )
    # Y <- array( Y, dim=c(dim(Y),1) )
    
    X = X[,dim(X)[2]:1,,drop=F]
    Y = Y[,dim(Y)[2]:1,drop=F]
    
    list(X = X, Y = Y)
  })
  
  model <- eventReactive(input$go,{
    print_test = function(model){
      pred = predictr(model,model$X_test,real_output = T,hidden = T)
      message(paste0("Test set: target/predict - ",sum(bin2int(model$Y_test)),"/",sum(bin2int(pred[[length(pred)]]))))
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
                      epoch_function = c(epoch_annealing,epoch_print,print_test)
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
    data = matrix(0,ncol=input$time_dim,nrow=4)
    data[1:2,] = test_set()[[1]][input$test_sample,,]
    data[3,]   = test_set()[[2]][input$test_sample,]
    data[4,]   = model()$test_store[[input$test_epoch]][[length(hidden_dim)+1]][input$test_sample,]
    rownames(data) <- c("X1","X2","Y","pred")
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
    
    data = matrix(0,ncol=input$time_dim,nrow=4)
    data[1:2,] = test_set()[[1]][input$test_sample,,]
    data[3,]   = test_set()[[2]][input$test_sample,]
    data[4,]   = model()$test_store[[input$test_epoch]][[length(hidden_dim)+1]][input$test_sample,]
    rownames(data) <- c("X1","X2","Y","pred")
    
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
