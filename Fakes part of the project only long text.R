library(e1071)
library(tm)
library(wordcloud)
library(gmodels)
library(textstem)

fakedf <- read.csv("fake_job_postings.csv", stringsAsFactors = FALSE)
head(fakedf)

#--- Filling the missing data
fakedf$employment_type[fakedf$employment_type == ""] <- "NoInfo" 
fakedf$required_experience[fakedf$required_experience == ""] <- "NoInfo"
fakedf$required_education[fakedf$required_education == ""] <- "NoInfo"
fakedf$industry[fakedf$industry == ""] <- "NoInfo"
fakedf$function.[fakedf$function. == ""] <- "NoInfo"


#--- Removing the job_id column ------------------------
fakedf <- fakedf[,-c(1)]
#summary(fakedf)

#--- Adding factor columns
fakedf$telecom_fact <- as.factor(fakedf$telecommuting)
fakedf$logo_fact <- as.factor(fakedf$has_company_logo)
fakedf$quest_fact <- as.factor(fakedf$has_questions)
fakedf$empl_typ_fact <- as.factor(fakedf$employment_type)
fakedf$req_exp_fact <- as.factor(fakedf$required_experience)
fakedf$req_ed_fact <- as.factor(fakedf$required_education)
fakedf$indus_fact <- as.factor(fakedf$industry)
fakedf$fraud_fact <- as.factor(fakedf$fraudulent)

#--- Concatinated columns of company profile, description, requirements and benfits
fakedf$long_text <- paste(fakedf$company_profile, fakedf$description, fakedf$requirements, fakedf$benefits)

#--- Cleaning texts

corpus_long <- Corpus(VectorSource(fakedf$long_text))

#-- Cleaning of long_text
clean_long <- tm_map(corpus_long, tolower)
clean_long <- tm_map(clean_long, removeNumbers)
clean_long <- tm_map(clean_long, removePunctuation)
clean_long <- tm_map(clean_long, removeWords, stopwords("en"))
clean_long <- tm_map(clean_long, removeWords, stopwords("SMART"))
clean_long <- tm_map(clean_long, stripWhitespace)
clean_long <- tm_map(clean_long, content_transformer(lemmatize_words))

#The following code defines a convert_counts() function to convert counts to factors
convert_counts <- function(x) { 
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x) }

desc_dtm_all <- DocumentTermMatrix(clean_long)
fake_dict_all <- findFreqTerms(desc_dtm_all, 250)
fake_all <- DocumentTermMatrix(clean_long, list(dictionary = fake_dict_all))
fake_all2 <- apply(fake_all, MARGIN = 2, convert_counts)

#--- Transformation of the fake_all2 matrix into a data frame
fake_all2_df <- as.data.frame(fake_all2)
#- Transform the data frame into factors
fake_all2_df <- as.data.frame(unclass(fake_all2_df),stringsAsFactors=TRUE)

#--- new data frame only with the desired data
new_df <- data.frame(fakedf$telecom_fact, fakedf$logo_fact, fakedf$quest_fact, fakedf$empl_typ_fact, fakedf$req_exp_fact, fakedf$req_ed_fact, fakedf$indus_fact, fakedf$fraud_fact)

#-- merging of new_df and fake_all2_df into one data frame
final_df <- cbind(new_df, fake_all2_df)
str(final_df)

#--- Creation and testing of the model 
fake_classifier2 <- naiveBayes(fakedf.fraud_fact ~., data = final_df)
fake_test_pred2 <- predict(fake_classifier2, final_df)

CrossTable(fake_test_pred2, final_df$fakedf.fraud_fact,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

#-- Testing using the train and test data --------------------------------------------------
#- New training and test data
fakedf_train_final <- final_df[1:14304, ]
fakedf_test_final <- final_df[14305:17880, ]


fake_classifier3 <- naiveBayes(fakedf.fraud_fact ~., data = fakedf_train_final)
fake_test_pred3 <- predict(fake_classifier2, fakedf_test_final)

CrossTable(fake_test_pred3, fakedf_test_final$fakedf.fraud_fact,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))