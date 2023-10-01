library(class)
library(e1071)
library(tm)
library(gmodels)
# For decision tree
library(tree)
library(ISLR2)
library(rpart)

diabdf <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv", stringsAsFactors = FALSE)
head(diabdf)

#--- Treating the categorical data -------
# diabdf$Diabetes_binary <- as.factor(diabdf$Diabetes_binary) decision tree works better with this as numeric
diabdf$HighBP <- as.factor(diabdf$HighBP)
diabdf$HighChol <- as.factor(diabdf$HighChol)
diabdf$CholCheck <- as.factor(diabdf$CholCheck)
diabdf$Smoker <- as.factor(diabdf$Smoker)
diabdf$Stroke <- as.factor(diabdf$Stroke)
diabdf$HeartDiseaseorAttack <- as.factor(diabdf$HeartDiseaseorAttack)
diabdf$PhysActivity <- as.factor(diabdf$PhysActivity)
diabdf$Fruits <- as.factor(diabdf$Fruits)
diabdf$Veggies <- as.factor(diabdf$Veggies)
diabdf$HvyAlcoholConsump <- as.factor(diabdf$HvyAlcoholConsump)
diabdf$AnyHealthcare <- as.factor(diabdf$AnyHealthcare)
diabdf$NoDocbcCost <- as.factor(diabdf$NoDocbcCost)
diabdf$GenHlth <- as.factor(diabdf$GenHlth)
#diabdf$MentHlth <- as.factor(diabdf$MentHlth)
#diabdf$PhysHlth <- as.factor(diabdf$PhysHlth)
diabdf$DiffWalk <- as.factor(diabdf$DiffWalk)
diabdf$Sex <- as.factor(diabdf$Sex)
#diabdf$Age <- as.factor(diabdf$Age)
#diabdf$Education <- as.factor(diabdf$Education)
#diabdf$Income <- as.factor(diabdf$Income)

diab_train <- diabdf[c(1:28277, 35347:63624),]
diab_test <- diabdf[c(28278:35346, 63625:70692),]

diab_train_1 <- diab_train[, -c(5, 15, 16, 17, 20,21,22)]
diab_test_1 <- diab_test[, -c(5, 15, 16, 17, 20,21,22)]
#------------------ Naïve Bayes section ------------------------------


diab_classifier <- naiveBayes(diab_train, diab_train$Diabetes_binary) #Model building
diab_test_pred <- predict(diab_classifier, diab_test) #Model prediction

CrossTable(diab_test_pred, diab_test$Diabetes_binary,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual')) #Model testing

#------------------ Decision tree section ------------------
#--- First tree with library "tree"
tree.diab <- tree(Diabetes_binary ~., diab_train_1)
summary(tree.diab)

plot(tree.diab)
text(tree.diab)

#-- Prediction -----------------------
diab_test_pred1 <- round(predict(tree.diab, diab_test_1), 0) #Model prediction

CrossTable(diab_test_pred1, diab_test_1$Diabetes_binary,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual')) #Model testing

#--- Second tree with library "rpart" ----------------------
tree.diab_2 <- rpart(Diabetes_binary ~., data=diab_train_1, method= "class")

printcp(tree.diab_2) # display the results
plotcp(tree.diab_2) # visualize cross-validation results
summary(tree.diab_2) # detailed summary of splits

# plot tree
plot(tree.diab_2, uniform=TRUE,
     main="Classification Tree for Diabetes")
text(tree.diab_2, use.n=TRUE, all=TRUE, cex=.8)

diab_test_pred2 <- predict(tree.diab_2, diab_test_1, type="class") #Model prediction

diab_test_pred2table <- table(predict(tree.diab_2, diab_test_1, type="class"), diab_test_1$Diabetes_binary) #Model prediction

CrossTable(diab_test_pred2, diab_test_1$Diabetes_binary,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual')) #Model testing

#---------------- Support Vector machines -----------------
svm_1 <- svm(Diabetes_binary ~., data = diab_train_1, kernel = "radial", gamma = 1, Cost = 1)
svm_1_pred <- predict(svm_1, diab_test_1)

CrossTable(svm_1_pred, diab_test_1$Diabetes_binary,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual')) #Model testing

svm_2 <- svm(Diabetes_binary ~., data = diab_train_1, kernel = "linear", Cost = 1, scale = FALSE)
svm_2_pred <- predict(svm_2, diab_test_1)

CrossTable(svm_2_pred, diab_test_1$Diabetes_binary,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual')) #Model testing

#-----------------------------Logistic regression----------------------------------------
#-- Building "blind" model
lrm_0 <- glm(Diabetes_binary ~ ., data=diabdf, family='binomial')
summary(lrm_0)

par(mfrow=c(2,2)) 
plot(lrm_0)

lrm <- glm(Diabetes_binary ~ .-Smoker- PhysActivity- Fruits- AnyHealthcare- NoDocbcCost, data=diabdf, family='binomial')
summary(lrm)

par(mfrow=c(2,2)) 
plot(lrm)

#-- Accuracy calc -------------------------------------
actual <- diabdf$Diabetes_binary
predicted <- round(fitted(lrm))
xt <- xtabs(~actual+predicted)
xt
accuracy <- (xt[1,1]+xt[2,2])/sum(xt)
accuracy

#-- Model without first three outliers
diabdf_out <- diabdf[-c(27802,28880, 30469),]

lrm_1 <- glm(Diabetes_binary ~ .-Smoker- PhysActivity- Fruits- AnyHealthcare- NoDocbcCost, data=diabdf_out, family='binomial')
summary(lrm_1)

par(mfrow=c(2,2)) 
plot(lrm_1)

#-- Accuracy calculation for model without first outliers -------------------------------------
actual_1 <- diabdf_out$Diabetes_binary
predicted_1 <- round(fitted(lrm_1))
xt_1 <- xtabs(~actual_1+predicted_1)
xt_1
accuracy_1 <- (xt_1[1,1]+xt_1[2,2])/sum(xt_1)
accuracy_1