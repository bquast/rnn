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

b2i <- function(binary)
  packBits(as.raw(rev(c(rep(0, 32-length(binary) ), binary))), 'integer')

binary_add_data <- function(seed,time_dim,sample_dim){
  # replicable
  set.seed(seed)
  
  # create training numbers
  X1 = sample(0:(2^(time_dim-1)-1), sample_dim, replace=TRUE)
  X2 = sample(0:(2^(time_dim-1)-1), sample_dim, replace=TRUE)
  
  # create training response numbers
  Y <- X1 + X2
  
  # convert to binary
  X1 <- int2bin(X1,time_dim)
  X2 <- int2bin(X2,time_dim)
  Y  <- int2bin(Y,time_dim)
  
  # Create 3d array: dim 1: samples; dim 2: time; dim 3: variables.
  X <- array( c(X1,X2), dim=c(dim(X1),2) )
  Y <- array( Y, dim=c(dim(Y),1) ) 
  
  X = X[,dim(X)[2]:1,,drop=F]
  Y = Y[,dim(Y)[2]:1,,drop=F]
  
  list(X = X, Y = Y)
}

cosinus_sinus_data <- function(seed,time_dim,sample_dim,noise){
  # synthetic time serie prediction
  set.seed(seed)
  X <- data.frame()
  Y <- data.frame()
  
  # Genereate a bias in phase
  bias_phase <- rnorm(sample_dim)
  # Generate a bias in frequency
  bias_frequency = runif(sample_dim,min=5,max=25)
  
  # Generate the noisy time series, cosinus for X and sinus for Y, with a random bias in phase and in frequency
  for(i in seq(sample_dim)){
    X <- rbind(X,sin(seq(time_dim)/bias_frequency[i]+bias_phase[i])+rnorm(time_dim,mean=0,sd=noise))
    Y <- rbind(Y,cos(seq(time_dim)/bias_frequency[i]+bias_phase[i]))
  }
  
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  # Normalize between 0 and 1 for the sigmoid
  X <- (X-min(X))/(max(X)-min(X))
  Y <- (Y-min(Y))/(max(Y)-min(Y))
  
  X <- array(X,dim = c(dim(X),1))
  Y <- array(Y,dim = c(dim(Y),1))

  list(X = X, Y = Y)
}


server <- function(input, output,session) {
  output$hidden <- renderUI({
    x <- list()
    for(i in seq(input$layers)){
      x[[i]] <- numericInput(inputId = paste0('hidden',i),label = paste0('Number of unit in the layer number ',i),value = 10)
    }
    return(tagList(x))
  })
  
  train_set <- reactive({
    # if(input$problem == "binary addition"){
      data = binary_add_data(seed = 1,time_dim = input$time_dim_binary,sample_dim = input$sample_dim_train_binary)
    # }
    # if(input$problem == "cosinus sinus"){
      # data = cosinus_sinus_data(seed = 1,time_dim = input$time_dim_cos,sample_dim = input$sample_dim_train_cos,noise = input$noise_cos)
    # }
    data
  })
  
  test_set <- reactive({
    # if(input$problem == "binary addition"){
      data = binary_add_data(seed = 2,time_dim = input$time_dim_binary,sample_dim = input$sample_dim_test_binary)
    # }
    # if(input$problem == "cosinus sinus"){
      # data = cosinus_sinus_data(seed = 2,time_dim = input$time_dim_cos,sample_dim = input$sample_dim_test_cos,noise = input$noise_cos)
    # }
    data
  })
  
  model <- eventReactive(input$go,{
    print_test = function(model){
      pred = predictr(model,model$X_test,real_output = F,hidden = T)
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
                      epoch_model_function = c(print_test)
                      )
      return(model)
    })
  })
  
  
  output$error <- renderPlot({
    error = colMeans(model()$error)
    par(mfrow=c(2,1))
    plot(error,type="l",xlab="epoch",ylab = "error",main = "training set")
    error = colMeans(model()$test_error)
    plot(error,type="l",xlab="epoch",ylab = "error",main = "test set")      
  })
  
  table_observe <- eventReactive(input$go,{
    hidden_dim = c()
    for(i in seq(input$layers)){
      hidden_dim[i] <- input[[paste0('hidden',i)]]
    }
    data = t(array(test_set()[[1]][input$test_sample,,],dim=c(dim(test_set()[[1]])[2:3])))
    data = rbind(data,test_set()[[2]][input$test_sample,,],model()$test_store[[input$test_epoch]][[length(hidden_dim)+1]][input$test_sample,,])
    # data = rbind(data, test_set()[[2]][input$test_sample,,])
    # data = rbind(data,model()$test_store[[input$test_epoch]][[length(hidden_dim)+1]][input$test_sample,,])
    rownames(data) <- c(paste0("X",seq(dim(test_set()[[1]])[3]),collapse = ""),"Y","pred")
    for(i in seq(length(hidden_dim))){
      truc = array(model()$test_store[[input$test_epoch]][[i]][input$test_sample,,],dim=c(hidden_dim[i],input$time_dim))
      rownames(truc) = paste0("layer ",i," - hidden ",seq(hidden_dim[i]))
      data = rbind(data,truc)
    }
    data
  })
  
  output$table_observe <- renderTable({
    table_observe()
  }) #digits = 0
  output$plot_observe <- renderPlot({
    hidden_dim = c()
    for(i in seq(input$layers)){
      hidden_dim[i] <- input[[paste0('hidden',i)]]
    }
    par(mfrow=c(input$layers+1,1))
    matplot(t(table_observe()[1:4,]),type="l")
    for(i in seq(length(hidden_dim))){
      truc = array(model()$test_store[[input$test_epoch]][[i]][input$test_sample,,],dim=c(hidden_dim[i],input$time_dim))
      rownames(truc) = paste0("layer ",i," - hidden ",seq(hidden_dim[i]))
      matplot(t(truc),type="l",xlab="time steps",ylab = "synapse values",main = paste0("layer ",i))
    }
  })
}

ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
        h3("data"),
        selectizeInput("problem","learning task",choices = c("binary addition","cosinus sinus")),
        conditionalPanel(
          'input.problem == "binary addition"',
          numericInput("time_dim_binary","time dimension",8),
          numericInput("sample_dim_train_binary","training sample dimension",7000),
          numericInput("sample_dim_test_binary","testing sample dimension",7000)
            ),
        conditionalPanel(
          'input.problem == "cosinus sinus"',
          numericInput("time_dim_cos","time dimension",200),
          numericInput("sample_dim_train_cos","training sample dimension",100),
          numericInput("sample_dim_test_cos","testing sample dimension",10),
          numericInput("noise_cos","noise apply on the input",0.2)
        ),
        
        h3("network"),
        numericInput("layers","number of hidden layers",2),
        uiOutput("hidden"),
        numericInput("learningrate","learningrate",0.1),
        numericInput("batchsize","batchsize",100),
        numericInput("numepochs","numepochs",5),
        numericInput("momentum","momentum",0.5),
        checkboxInput("use_bias","use_bias",T),
        numericInput("learningrate_decay","learningrate_decay",1),
        actionButton("go","Train")
        
    ),
    mainPanel(
      plotOutput("error"),
      numericInput("test_sample","test sample index to look at",1),
      numericInput("test_epoch","epoch to look at",1),
      tableOutput("table_observe"),
      plotOutput("plot_observe")
      )
  )
)

shinyApp(ui = ui, server = server)
