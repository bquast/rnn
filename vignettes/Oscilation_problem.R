## R script for oscillation LSTM test, not learning yet

set.seed(1)

sample_dim = 10000 # number of sampels
time_dim = 10 # time dimension
variable_dim = 1 # number of variables, we will see later to make it more complex

event_proba = 0.1# percent of 1 in the time series

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

l = create_data_set(sample_dim,time_dim,variable_dim,event_proba,simple = T)
X = l[[1]]
Y = l[[2]]
print(dim(X))
print(dim(Y))

set.seed(1)

print_test = function(model){
  message(paste0("Trained epoch: ",model$current_epoch," - Learning rate: ",model$learningrate))
  message(paste0("Epoch error: ",colMeans(model$error)[model$current_epoch]))
  pred = model$store[[length(model$store)]]
  n = sample(seq(nrow(pred)),1)
  for(i in seq(variable_dim)){
    print(paste("X",i,":", paste(X[n,,i], collapse = " ")))
    print(paste("Pred:", paste(round(pred[n,,i]), collapse = " ")))
    print(paste("True:", paste(Y[n,,i], collapse = " ")))
    # for(j in seq(model$hidden_dim[1])){
    #   print(paste(j,"H:", paste(round(model$store[[1]][n,,i,1]), collapse = " ")))
    #   print(paste(j,"z:", paste(round(model$store[[1]][n,,i,2]), collapse = " ")))
    #   print(paste(j,"r:", paste(round(model$store[[1]][n,,i,3]), collapse = " ")))
    #   print(paste(j,"h:", paste(round(model$store[[1]][n,,i,4]), collapse = " ")))
    # }
    print(paste("round error:",sum(abs(round(pred[,,i]) - Y[,,i])),"/",dim(Y)[1]*dim(Y)[2]))
    print(paste("perfect:",sum(apply(abind::abind(Y[,,i],round(pred[,,i]),along = 3),1,function(x){sum(x[,1] == x[,2]) == time_dim})),"/",sample_dim))
  }
  return(model)
}

# train the model
model <- trainr(Y=Y,
                X=X,
                learningrate   =  0.005, # best 0.05
                hidden_dim     =  c(4,4,4),# best c(4,4,4)
                batch_size     = 20, # best 20, not ok 50
                numepochs      =  5, # best 50
                momentum       =0, # best 0, not ok 0.5
                use_bias       = T, # best T
                update_rule = "adagrad",
                network_type = "rnn", # best rnn
                # clipping = 100000, # best 100000
                learningrate_decay = 0.9, # best 0.9
                epoch_function =  c(epoch_annealing,print_test)
                )

