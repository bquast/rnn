## R script for oscillation LSTM test, not learning yet

set.seed(1)

X_dim_1 = 7000 # number of sampels
X_dim_2 = 100 # time dimension
X_dim_3 = 1 # number of variables, we will see later to make it more complex


X_event_proba = 0.1 # percent of 1 in the time series

X1 = array(sample(0:1,size=X_dim_1*X_dim_2,replace=TRUE,prob = c(1-X_event_proba,X_event_proba)),dim=c(X_dim_1,X_dim_2))
Y1 = array(rep(0,X_dim_1*X_dim_2),dim=c(X_dim_1,X_dim_2))


## rolling sum function
rollSum <- function(x){
  y = c()
  for(i in seq(length(x))){
    y = c(y,sum(x[1:i]) %% 2)
  }
  return(y)
}

for(i in seq(X_dim_1)){
  Y1[i,] <- rollSum(X1[i,])
}

# View(rbind(X[1,],Y[1,]))

X1 <- array(X1,dim=c(dim(X1),1))
Y1 <- array(Y1,dim=c(dim(Y1),1))

set.seed(1)

# train the model
model <- trainr(Y=Y1,
                X=X1,
                learningrate   =  0.05,
                hidden_dim     =  c(4),
                batch_size     = 100,
                numepochs      =  500,
                momentum       =0.5,
                use_bias       = T,
                network_type = "lstm",
                clipping = 100000,
                learningrate_decay =1
                )