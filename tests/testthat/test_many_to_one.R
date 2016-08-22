# replicable
load("~/Dropbox/rnn/data/tlc_data.Rda")

set.seed(1)

X = aperm(tlc_data[[1]],c(2,1,3))
Y = aperm(tlc_data[[2]],c(2,1))

plot_function <- function(model){
  par(mfrow=c(1,3))
  DLC::raster(X)
  DLC::raster(Y)
  DLC::raster(round(model$store[[length(model$store)]][,,1][,1,drop=F]))
}

# train the model
model <- trainr(Y=Y,
                X=X,
                learningrate   =  0.01,
                hidden_dim     =  c(30),
                numepochs      =  1000,
                batch_size     = 10,
                momentum       =0.1,
                use_bias       = T,
                learningrate_decay = 1,
                many_to_one = T,
                network_type="lstm",
                clipsing = 1000,
                epoch_function = c(epoch_print,plot_function))


# expect_equal(sum(bin2int(B)), 888626)
# print(sum(bin2int(B)))
# print(sum(bin2int(A1))+sum(bin2int(A2)))


