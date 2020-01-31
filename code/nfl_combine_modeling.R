# start with jump-start code... no need to run the data prep code

# Here we read in the training and test files
# Read text columns as character strings and convert selected columns to factor variables.
# The training set will be used to create build and validation sets in cross-validation.
training <- read.csv(file = "training.csv", header = TRUE, stringsAsFactors = FALSE) 
 
# The test set will be set aside during the model development process and used
# and used only after the full model has been developed and tested.            
test <- read.csv(file = "test.csv", header = TRUE, stringsAsFactors = FALSE)   

# For consistency in team codes between training and test, set STL to LAR in training set.
for (i in seq(along = training$Tm)) {
    if (training$Tm[i] == 'STL') training$Tm[i] <- 'LAR'
    }

# Here we show how to set up a factor variable for the NFL team
training$Tm <- factor(training$Tm)                        
test$Tm <- factor(test$Tm) 

# Quick check of the data frames 
cat('\n Summary of Training Set\n')
print(summary(training))
cat('\n Summary of Test Set\n')
print(summary(test))  

#Missing Data Test
nulls <- data.frame(col=as.character(colnames(training)), pct_null=colSums(is.na(training))*100/(colSums(is.na(training))+colSums(!is.na(training))))
nulls

# Because the modeling function for this assignment may involve many steps,
# we set it up as a function. Input to the function includes definition of
# the training and test sets for an iteration/fold of cross-validation.
# Within the cross-validation procedure, training_input and test_input will be 
# subsets of the original full training set. At the very end of the model
# development process, of course, training_input and test_input will be the full
# training and test sets, but that only comes at the very end of the process.
# The function returns the root mean-square error in test_input set. 

eval_model <- function(training_input, test_input) {
  qb.k.p.model <- lm(Pick ~ Ht + Wt + Shuttle + Vertical + BroadJump, data = training_input, na.action = 'na.omit')
  qb.k.p.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$Shuttle[i]) &&
        !is.na(test_input$Vertical[i]) && !is.na(test_input$BroadJump[i]))     
      qb.k.p.predict[i] <- predict.lm(qb.k.p.model, newdata = test_input[i,])
  
  rb.model <- lm(Pick ~ Ht + Wt + X40yd + X3Cone + BroadJump, data = training_input, na.action = 'na.omit')
  rb.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$X40yd[i]) &&
        !is.na(test_input$X3Cone[i]) && !is.na(test_input$BroadJump[i]))
      rb.predict[i] <- predict.lm(rb.model, newdata = test_input[i,]) 
  
  wr.model <- lm(Pick ~ Ht + Wt + X40yd + Vertical + X3Cone, data = training_input, na.action = 'na.omit')
  wr.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$X40yd[i]) &&
        !is.na(test_input$Vertical[i]) && !is.na(test_input$X3Cone[i]))
      wr.predict[i] <- predict.lm(wr.model, newdata = test_input[i,])
  
  fb.te.ls.model <- lm(Pick ~ Ht + Wt + Bench + Vertical + BroadJump, data = training_input, na.action = 'na.omit')
  fb.te.ls.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$Bench[i]) &&
        !is.na(test_input$Vertical[i]) && !is.na(test_input$BroadJump[i]))
      fb.te.ls.predict[i] <- predict.lm(fb.te.ls.model, newdata = test_input[i,])

  ot.model <- lm(Pick ~ Ht + Wt + Shuttle + X40yd + Bench, data = training_input, na.action = 'na.omit')
  ot.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$Shuttle[i]) &&
        !is.na(test_input$X40yd[i]) && !is.na(test_input$Bench[i]))
      ot.predict[i] <- predict.lm(ot.model, newdata = test_input[i,])
    
  og.c.dt.model <- lm(Pick ~ Ht + Wt + Shuttle + Bench + BroadJump, data = training_input, na.action = 'na.omit')
  og.c.dt.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$Shuttle[i]) &&
        !is.na(test_input$Bench[i]) && !is.na(test_input$BroadJump[i]))
      og.c.dt.predict[i] <- predict.lm(og.c.dt.model, newdata = test_input[i,])
  
  de.prolb.model <- lm(Pick ~ Ht + Wt + X3Cone + Vertical + BroadJump, data = training_input, na.action = 'na.omit')
  de.prolb.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$X3Cone[i]) &&
        !is.na(test_input$Vertical[i]) && !is.na(test_input$BroadJump[i]))
      de.prolb.predict[i] <- predict.lm(de.prolb.model, newdata = test_input[i,])
  
  ilb.tolb.model <- lm(Pick ~ Ht + Wt + Shuttle + X3Cone + BroadJump, data = training_input, na.action = 'na.omit')
  ilb.tolb.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$Shuttle[i]) &&
        !is.na(test_input$X3Cone[i]) && !is.na(test_input$BroadJump[i]))
      ilb.tolb.predict[i] <- predict.lm(ilb.tolb.model, newdata = test_input[i,])
  
  cb.s.model <- lm(Pick ~ Ht + Wt + Shuttle + X40yd + X3Cone, data = training_input, na.action = 'na.omit')
  cb.s.predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    if (!is.na(test_input$Ht[i]) && !is.na(test_input$Wt[i]) && !is.na(test_input$Shuttle[i]) &&
        !is.na(test_input$X40yd[i]) && !is.na(test_input$X3Cone[i]))
      cb.s.predict[i] <- predict.lm(cb.s.model, newdata = test_input[i,])
  
  # We are creating an ensemble or hybrid prediction by averaging all component
  # model predictions with non-missing values. Do this one player at a time.    
  response_predict <- rep(NA, times = nrow(test_input))
  for (i in seq(along = test_input$Player)) 
    response_predict[i] <- mean(c(qb.k.p.predict[i], rb.predict[i], 
                                  wr.predict[i], fb.te.ls.predict[i], ot.predict[i],
                                  og.c.dt.predict[i], de.prolb.predict[i], ilb.tolb.predict[i], 
                                  cb.s.predict[i]), na.rm = TRUE)
  
  response_actual <- test_input$Pick
  ensemble_data_frame <- data.frame(qb.k.p.predict, rb.predict, wr.predict, fb.te.ls.predict, ot.predict,
                                    og.c.dt.predict, de.prolb.predict, ilb.tolb.predict, cb.s.predict,
                                    response_predict, response_actual)
  
  # To check calculations, we can examine the first rows of the ensemble_data_frame
  cat('\nFirst and last six rows of ensemble_data_frame\n')
  print(head(ensemble_data_frame)) 
  cat(' . . . \n')
  print(tail(ensemble_data_frame))        
  
  # compute and return root mean-square error in test_input
  sqrt(mean((response_predict - response_actual)^2, na.rm = TRUE))
}

# Whatever model is used for prediction, we want it to do better than a null model
# that predicts the mean response value for every player. Null model is like no model.
null_model <- function(training_input, test_input) {
  # for demonstration purposes we show what would be the prediction 
  # of a null model... predicting the mean Pick for every player in test_input
  response_predict <- mean(test_input$Pick)
  response_actual <- test_input$Pick
  # compute and return root mean-square error in test_input
  sqrt(mean((response_predict - response_actual)^2))
}

# Cross-validation work
library(cvTools)
set.seed(9999)  # for reproducibility   
nfolds <- 10                  

study_folds <- cvFolds(nrow(training), K = nfolds, type = 'consecutive')

cv_model_results <- numeric(nfolds)  # initialize array to store fold model results
cv_null_results <- numeric(nfolds)  # initialize array to store fold null results
for (ifold in 1:nfolds) {
  cat('\nWorking on fold ', ifold, '\n')
  this_fold_test_data <- training[study_folds$which == ifold,]
  this_fold_training_data <- 
    training[study_folds$which != ifold,]
  # fit model and get root mean-square error for this iteration   
  cv_model_results[ifold] <- eval_model(training_input = this_fold_training_data,
                                        test_input = this_fold_test_data)    
  cv_null_results[ifold] <- null_model(training_input = this_fold_training_data,
                                       test_input = this_fold_test_data)    
}
cat('\n', 'Cross-validation My Model Average Root Mean-Square Error:', 
    mean(cv_model_results))  

cat('\n', 'Cross-validation No Model Average Root Mean-Square Error:', 
    mean(cv_null_results))

cv_model_results_mean <- mean(cv_model_results)
cv_null_results_mean <- mean(cv_null_results)

plot(cv_model_results, xlab = "Model Results", ylab = "RMSE", main = "Model Performance", type = "p", col = "blue")
points(cv_null_results, col = "red")
abline(h = cv_model_results_mean, col = "blue")
abline(h = cv_null_results_mean, col = "red")
legend("topright", legend=c("Evaluation Model RMSE Values", "Null Model RMSE Values"), col=c("blue", "red"), pch = 1, bty= "n", cex=0.8)
legend("bottomleft", legend=c("Evaluation Model Average RMSE", "Null Model Average RMSE"), col=c("blue", "red"),lty=c(1,1),bty= "n", cex=0.8)