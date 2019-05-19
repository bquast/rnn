# binary addition demo for rnn package
# just an app.R file to keep it simple, hopefully, it will stay this way

library(rnn)
library(shiny)
library(sigmoid)

create_data_set = function(sample_dim,time_dim,variable_dim,event_proba,simple=F){
  if(simple == F){
    X = array(sample(0:1,size=sample_dim*time_dim*variable_dim,replace=TRUE,prob = c(1-event_proba,event_proba)),dim=c(sample_dim,time_dim,variable_dim))
  }else{
    X = array(0,dim=c(sample_dim,time_dim,variable_dim))
    for(i in seq(sample_dim)){
      for(j in seq(variable_dim)){
        X[i,sample(seq(time_dim),2),j] = 1}
    }
  }
  
  rollSum <- function(x){
    y = c()
    for(i in seq(length(x))){
      y = c(y,sum(x[1:i]) %% 2)
    }
    return(y)
  }
  Y = X
  for(i in seq(variable_dim)){
    for(j in seq(sample_dim)){
      Y[j,,i] <- rollSum(X[j,,i])
    }
  }
  return(list(X,Y))
}


oscillationUI <- function(id){
  ns = NS(id)
  sidebarLayout(
    sidebarPanel(width = 3,
      h3("data"),
      actionButton(ns("go"),"Train"),
      p("This artificial dataset consist of 1 and 0, a rolling sum is applied on the time series to produce the output and modulo 2 is applied on the numbers, leading to 1 and 0. So the output will switch at each event."),
      numericInput(ns("time_dim"),"time dimension",10),
      checkboxInput(ns("simple_dataset"),"should the time series only have 2 events (over ride the event proba)",T),
      numericInput(ns("event_proba"),"probability of event in the time series",0.1),
      numericInput(ns("sample_dim_train"),"training sample dimension",1000),
      numericInput(ns("sample_dim_test"),"testing sample dimension",100),
      h3("network"),
      numericInput(ns("layers"),"number of hidden layers",3),
      uiOutput(ns("hidden")),
      numericInput(ns("learningrate"),"learningrate",0.05),
      numericInput(ns("batchsize"),"batchsize",20),
      numericInput(ns("numepochs"),"numepochs",5),
      numericInput(ns("momentum"),"momentum",0),
      checkboxInput(ns("use_bias"),"use_bias",T),
      numericInput(ns("learningrate_decay"),"learningrate_decay",0.9)
      
    ),
    mainPanel(
      # img(src="binary.gif"),
      plotOutput(ns("error")),
      numericInput(ns("test_sample"),"test sample index to look at",1),
      numericInput(ns("test_epoch"),"epoch to look at",1),
      tableOutput(ns("table_observe"))
      
    )
  )
}

oscillationServer <- function(input, output,session) {
  output$hidden <- renderUI({
    ns <- session$ns
    x <- list()
    for(i in seq(input$layers)){
      x[[i]] <- numericInput(inputId = ns(paste0('hidden',i)),label = paste0('Number of unit in the layer number ',i),value = 4)
    }
    return(tagList(x))
  })
  
  train_set <- reactive({
    # replicable
    create_data_set(input$sample_dim_train, input$time_dim, variable_dim = 1, input$event_proba, input$simple_dataset)
  })
  
  test_set <- reactive({
    # replicable
    set.seed(2)
    create_data_set(input$sample_dim_test, input$time_dim, variable_dim = 1, input$event_proba, input$simple_dataset)
  })
  
  model <- eventReactive(input$go,{
    print_test = function(model){
      message(paste0("Trained epoch: ",model$current_epoch," - Learning rate: ",model$learningrate))
      message(paste0("Epoch error: ",colMeans(model$error)[model$current_epoch]))
      pred = predictr(model,model$X_test,real_output = F,hidden = T)
      pred_last = pred[[input$layers + 1]]
      n = sample(seq(nrow(pred_last)),1)
      for(i in seq(1)){
        print(paste("X",i,":", paste(model$X_test[n,,i], collapse = " ")))
        print(paste("Pred:", paste(round(pred_last[n,,i]), collapse = " ")))
        print(paste("True:", paste(model$Y_test[n,,i], collapse = " ")))
        print(paste("perfect:",sum(apply(abind::abind(model$Y_test[,,i],round(pred_last[,,i]),along = 3),1,function(x){sum(x[,1] == x[,2]) == input$time_dim})),"/",input$sample_dim_test))
      }
      if(is.null(model$test_error)){model$test_error = array(0,dim = c(dim(model$Y_test)[1],model$numepochs))}
      model$test_error[,model$current_epoch] <- apply(model$Y_test - pred_last,1,function(x){sum(abs(x))})
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
    data[1,] = test_set()[[1]][input$test_sample,,]
    data[2,]   = test_set()[[2]][input$test_sample,,]
    data[3,]   = model()$test_store[[input$test_epoch]][[length(hidden_dim)+1]][input$test_sample,,]
    rownames(data) <- c("X","Y","pred")
    for(i in seq(length(hidden_dim))){
      truc = array(model()$test_store[[input$test_epoch]][[i]][input$test_sample,,],dim=c(hidden_dim[i],input$time_dim))
      rownames(truc) = paste0("layer ",i," - hidden ",seq(hidden_dim[i]))
      data = rbind(data,truc)
    }
    
    data
  },digits = 0)
  
}
