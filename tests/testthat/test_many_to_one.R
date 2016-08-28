# replicable
load("~/Dropbox/rnn/data/tlc_data.Rda")

set.seed(1)

X = aperm(tlc_data[[1]],c(2,1,3))
X = X[,124:10,,drop=F]
Y = aperm(tlc_data[[2]],c(2,1))
Y = Y[,124:10,drop=F]

plot_function <- function(model){
  par(mfrow=c(1,3),mar=c(0,0,0,0),xaxt="n",yaxt="n",xaxs="i",yaxs="i")
  DLC::raster(X)
  DLC::raster(Y)
  DLC::raster(model$store[[length(model$store)]])
}

# train the model
model <- trainr(Y=X[,115:1,],
                X=X,
                learningrate   =  0.001,
                hidden_dim     =  c(16,16),
                numepochs      =  1000,
                batch_size     = 10,
                momentum       =0.1,
                use_bias       = F,
                learningrate_decay = 1,
                # many_to_one = T,
                network_type="gru",
                epoch_function = c(epoch_print,plot_function))





