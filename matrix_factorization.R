# this is the main file to call functions to train models and run cross-validations
# set working directory
require(readr)
require(dplyr)

source('./util.R')
# dataset is too big so not included in the repository
train.dat <- read_csv("Reviews_train.csv")
test.dat <- read_csv("Reviews_test.csv")

# compute baseline rmse on training set and on test set
ans <- ComputeBaseline(train.dat, test.dat)
ans$train.rmse
ans$test.rmse

# ComputeSGDRatingOnly runs stochastic gradient descent and computes training 
# and test rmse at different iterations
ans <- ComputeSGDRatingOnly(train.dat[, c(2, 3, 7)], test.dat[, c(2, 3, 7)], 
                            latent.dim=10, max.iter=1, init.eta=0.02, 
                            step.size='fixed', decay.pow=0.25, rmse.interval=1, 
                            model.type='intercept', lambda=2)

# compare models with intercept and different combinations of lambdas and 
# latent dimensionsthrough cross-validations                                                                                                                
cv.ans.int <- CVSelectModel(train.dat, K=3, max.iter=30,
                        lambdas=c(0.5, 1, 2, 4, 0.5, 1, 2, 4, 0.5, 1, 2, 4), 
                        latent.dims=c(5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20), 
                        model.types=rep('intercept', 12))

# compare models without intercept and different combinations of lambdas and 
# latent dimensionsthrough cross-validations                                                                                                                
cv.ans.noint <- CVSelectModel(train.dat, K=3, max.iter=5,
                        lambdas=c(0.5, 1, 2, 4, 0.5, 1, 2, 4), 
                        latent.dims=c(5, 5, 5, 5, 10, 10, 10, 10), 
                        model.types=rep('no intercept', 8))

# output results
#write_csv(cv.ans.int, '../../experiment_results/result_1_5_2017_mf/some_cv_rmse.csv')

#write_csv(data.frame(train_rmse=ans$train.rmse,
#                      test_rmse=ans$test.rmse), 
#                      '../../experiment_results/result_1_1_2017_mf/train_test_rmse.csv')

