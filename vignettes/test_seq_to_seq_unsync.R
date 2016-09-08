# this script shows how to use the seq_to_seq option as well as the format of the output, 
# this do not learn yet or at least for this user case and with those options but the proof of concept is here

library(rnn)

time_dim = 15
sample_dim = 10000
event_proba = 0.1

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

set.see(1)
l = create_data_set(sample_dim = sample_dim,time_dim =time_dim,variable_dim = 1,event_proba = event_proba,F)


# train the model
model <- trainr(Y=l[[2]][,time_dim,,drop=F],
                X=l[[1]],
                learningrate   =  0.05,
                hidden_dim     =  c(4,4),
                numepochs      =  10,
                batch_size     = 10,
                momentum       =0,
                use_bias       = T,
                learningrate_decay = 1,
                seq_to_seq_unsync = T,
                network_type="rnn",
                epoch_function = c(epoch_print))



str(predictr(model,X=l[[1]]))

