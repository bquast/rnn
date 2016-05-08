# download exchange rate data
library(quantmod)
normalize <- function(x){x<-x-min(x);return(x/max(x))}
library(sigmoid)
library(rnn)
library(magrittr)

## tomorrows value of EUR/USD
getFX("EUR/USD",from="1998-12-15", to = '2003-12-15')
fEURUSD <- EURUSD

## todays values
getFX("CHF/USD",from="1998-12-15", to = '2003-12-15')
getFX("GBP/USD",from="1998-12-15", to = '2003-12-15')
getFX("JPY/USD",from="1998-12-15", to = '2003-12-15')
getFX("EUR/USD",from="1998-12-15", to = '2003-12-15')

# put in matrix form
mEURUSD <- matrix(EURUSD, nrow = 1) %>% normalize
mCHFUSD <- matrix(CHFUSD, nrow = 1) %>% normalize
mGBPUSD <- matrix(GBPUSD, nrow = 1)%>% normalize
mJPYUSD <- matrix(JPYUSD, nrow = 1)%>% normalize

# stack matrices
X <- array(c(mEURUSD, mCHFUSD, mGBPUSD, mJPYUSD), dim=c(1,1827,4))
# save(X,file="inst/shinyapps/finance_demo/www/X.Rdata")

X.train <- X[,1:1000,]
X.train <- array(X.train,dim=c(1,1000,4))

y.train <- matrix(X[,2:1001,1],ncol=1000)

set.seed(1)
# train model
model <- trainr(X = X.train,
                Y = y.train,
                learningrate = 0.01,
                numepochs = 100,
                hidden_dim = 10)

X.test <- X[,2:dim(X)[2],]
X.test <- array(X.test,dim=c(1,dim(X)[2]-1,4))

y.test <- matrix(X[,2:dim(X)[2],1],ncol=dim(X)[2]-1)
# prediction
y.pred <- predictr(model, X.test)

# plot prediction
plot(y.test[1,],yaxt="n",type="l",col="green")
par(new=T)
plot(y.pred[1,],yaxt="n",type="l",col="red")
abline(v=1000)
legend("topright",col = c("green","red"),pch=1,legend = c("target","prediction"))
