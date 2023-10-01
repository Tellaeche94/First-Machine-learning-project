library(class)
library(gmodels)

diamdf <- read.csv("DiamondsPrices.csv", stringsAsFactors = FALSE)
head(diamdf)

#--- Data manipulation -------------------------
diamdf$cutFact <- as.factor(diamdf$cut)
diamdf$colFact <- as.factor(diamdf$color)
diamdf$claFact <- as.factor(diamdf$clarity)

diamdf$cutInt <- as.numeric(diamdf$cutFact)
diamdf$colInt <- as.numeric(diamdf$colFact)
diamdf$claInt <- as.numeric(diamdf$claFact)

#- For testing in the log transformation
diamdf$xplus <- diamdf$x + 1
diamdf$yplus <- diamdf$y + 1
diamdf$zplus <- diamdf$z + 1

#----------------Multiple Linear regression section---------------------------

#--- Function to create a histogram to compare with a normal distribution ---
histogram <- function(x)
{ 
  title <- paste(deparse(substitute(x), 500), collapse="\n") 
  sdx <- sd(x)
  mx <- mean(x)
  hist(x, prob=TRUE, 
       main=paste("Histogram of ",title),
       xlim=c(mx-3*sdx, mx+3*sdx))
  curve(dnorm(x, mean=mx, sd=sdx), col='red', lwd=3, add=TRUE)
}

par(mfrow=c(1,1))
histogram(diamdf$price) 
histogram(1/(diamdf$price))
histogram(log(diamdf$price))
histogram(sqrt(diamdf$price))

round(cor(diamdf[,c(7, 1, 5, 6, 8:10)]), 2) #Looking at the correlation it can be observed that the variables with the highest importance for the model will be the carat, x, y and z columns

#--- Creation of model ------------------
MLRmodel <- lm(price ~ carat+ x+ y+ z+ cutFact+ colFact +claFact, data = diamdf)
summary(MLRmodel)

par(mfrow=c(2,2))
plot(MLRmodel)

#-- Model with log transformation ---------------

MLRmodel1 <- lm(log(price) ~ log(carat)+ log(xplus)+ log(yplus)+ log(zplus)+ cutFact+ colFact + claFact, data = diamdf)
summary(MLRmodel1)

par(mfrow=c(2,2))
plot(MLRmodel1)

#- Checking the accuracy of the model with log transformation
veirf_tab <- data.frame(exp(MLRmodel1$fitted.values), diamdf$price)
1-mean(abs(veirf_tab$diamdf.price - veirf_tab$exp.MLRmodel1.fitted.value)/veirf_tab$diamdf.price)

#----------------KNN section--------------------------------------------------
#The objective of the KNN model will be to predict the cut classification of the entries

prop.table(table(diamdf$claFact))

normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }

diamdf_n <- as.data.frame(lapply(diamdf[c(1, 5:10, 14:16)], normalize))

summary(diamdf_n)

diamdf_train <- diamdf_n[1:43152, ] #80% of the data for training
diamdf_test <- diamdf_n[43153:53940, ] #20% of the data for testing

diamdf_train_labels <- diamdf[1:43152, 14]
diamdf_test_labels <- diamdf[43153:53940, 14]

diamdf_test_pred <- knn(train = diamdf_train, test = diamdf_test, cl = diamdf_train_labels, k=21)

CrossTable(x = diamdf_test_labels, y = diamdf_test_pred, prop.chisq=FALSE)

# Try Multiple k values
accuracy = function(actual, predicted) {
  mean(actual == predicted)
}

k_to_try = seq(1,by=2, len=50)
acc_k = rep(x = 0, times = length(k_to_try))

for(i in seq_along(k_to_try)) {
  diamdf_test_pred = knn(train = diamdf_train, 
                       test = diamdf_test, 
                       cl = diamdf_train_labels, 
                       k = k_to_try[i])
  acc_k[i] = accuracy(diamdf_test_labels, diamdf_test_pred)
  if (i == 21){
    CrossTable(x = diamdf_test_labels, y = diamdf_test_pred, prop.chisq=FALSE)
  }
}

# plot accuracy vs choice of k
plot(acc_k, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "classification accuracy",
     main = "Accuracy vs Neighbors")
# add lines indicating k with best accuracy
abline(v = which(acc_k == max(acc_k)), col = "darkorange", lwd = 1.5)
# add line for max accuracy seen
abline(h = max(acc_k), col = "grey", lty = 2)

diamdf_test_pred <- knn(train = diamdf_train, test = diamdf_test, cl = diamdf_train_labels, k=3)

CrossTable(x = diamdf_test_labels, y = diamdf_test_pred, prop.chisq=FALSE)
