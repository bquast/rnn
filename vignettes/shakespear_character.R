### create the data set

# read the doenloaded file

TEXTFILE = "data/pg100.txt" # download in the data folder (if no data folder, change the string or create the folder)
if (!file.exists(TEXTFILE)) {
     download.file("https://www.gutenberg.org/cache/epub/100/pg100.txt", destfile = TEXTFILE)
 }
shakespeare = readLines(TEXTFILE)

# prepare the string
shakespeare = shakespeare[shakespeare != ""] # remove empty string
shakespeare = gsub("[[:digit:]]","",shakespeare) # remove digit
shakespeare = gsub("[^[:^punct:].,]","",shakespeare,perl = T)# remove punctuation except dot and comma


shakespeare = tolower(shakespeare[130:length(shakespeare)]) # force lower case and remove the first 130 string that are not shakespear
max_length = max(nchar(shakespeare)) # store the longest string length

for(i in seq(length(shakespeare))){ # use paste0 to put every string the same length
  n = nchar(shakespeare[i])
  if(n < max_length){
    shakespeare[i] = paste0(shakespeare[i],paste0(rep(" ",max_length - n),collapse = ""))
  }
}

# prepare the array
data <- lapply(shakespeare,strsplit,split="") # split the string character per character

data = unlist(do.call(rbind,data)) # could be better but at the end, we have a very long vector
data = kohonen::classvec2classmat(data) # transforn this vector into matrix of 1 and 0

name_char = dimnames(data)[[2]] # store the column names of the previous matrix

data = array(data,dim = c(max_length,length(shakespeare),dim(data)[2])) # coerce to array with given dimension
data = aperm(data,c(2,1,3)) # aperm because array fill by column and not by row (or the inverse, I never know), like option byrow in the matrix function.

dimnames(data) = list(NULL,NULL,name_char) # give the name_char for dimnames of the last dimension
View(data[6,,]) ## check it
shakespeare[6] ## check it

dim(data) 
# first dim should be the number of string keeped from shakespear, 
# second dim should be the longest string length, 
# third dim should be the length of unique character.

## subset because small computer

data = data[1:10000,,]

### train the model itself
## unfortunatly, not enough RAM to analyse this 4 giga bit dataset...
library(rnn)

epoch_print_bis = function(model){
  print(mean(abs(data[,c(2:max_length,max_length),] - round(model$store[[length(model$store)]]))))
  return(model)
}
# 
model = trainr(X = data,Y = data[,c(2:max_length,max_length),],
               hidden_dim = c(32,32),learningrate = 0.0001,
               use_bias = T,network_type = "rnn",batch_size = 5,numepochs = 100,
               epoch_function = c(epoch_print,epoch_print_bis))


### observe the result
model_bis  = model # store the model in an other object
model_bis$time_dim =1 # change the time dim to 1 to use the output as input of the next time step


for(j in name_char){ # for every character, predict the following
  truc = data[1,,,drop=F]
  truc[1,,] = 0
  truc[,1,j] = 1
  for(i in 1:(dim(truc)[2]-1)){
    truc[,i+1,] = predictr(model_bis,truc[,i,,drop=F])
  }
  View(round(truc[1,,]))
  print(paste0(kohonen::classmat2classvec(truc[1,,]),collapse = ""))
}
 

# nothing really good yet, we could try predicting after 5 characters, still it's doing soomething, he understood that h comes after t.

