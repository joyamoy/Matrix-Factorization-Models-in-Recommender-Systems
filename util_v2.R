# helper functions for training and cross-validating matrix factorization models
# define a set of functions for the experiment
require(readr)
require(dplyr)

# computes loss function (with regularization) and RMSE on both training and 
# test set
ComputeLossRMSE <- function(dat, U, V, bU, bV, mu, lambda, model.type) {
  sse <- 0
  num.dat <- nrow(dat)
  pb <- txtProgressBar(min = 0, max = num.dat, style = 3)
  for (r in 1 : num.dat) {
    user.id <- dat[r, 'UserId']
    product.id <- dat[r, 'ProductId']
    tmp.score <- dat[r, 'Score']
    if (model.type == "intercept") {
      pred.score <- t(U[user.id, ]) %*% V[product.id, ] + bU[user.id] + bV[product.id] + mu
    } else if (model.type == 'baseline') {
      pred.score <- bU[user.id] + bV[product.id] + mu
    } else {
      pred.score <- t(U[user.id, ]) %*% V[product.id, ]
    }
    sse <- sse + (tmp.score - pred.score)^2
    setTxtProgressBar(pb, r)
  }
  close(pb)
  rmse <- sqrt(sse / nrow(dat))
  if (model.type == "intercept") {
    loss <- sse + lambda * (sum(U^2) + sum(bU^2)) + lambda * (sum(V^2) + sum(bV^2))
  } else if (model.type == 'baseline') {
    loss <- NA
  } else {
    loss <- sse + lambda * sum(U^2) + lambda * sum(V^2)
  }
  return(list(rmse=rmse, loss=loss))
}

# implement baseline estimator (two-way anova, no interaction between user and product)
ComputeBaseline <- function(train.dat, test.dat) {
  train.dat <- as.data.frame(train.dat)
  test.dat <- as.data.frame(test.dat)
  # make sure product and user levels are consistent between train set and test set
  # and with bU, bV, U, V names
  product.levels <- unique(train.dat[, 'ProductId'])
  user.levels <- unique(train.dat[, 'UserId'])
  train.dat[, 'ProductId'] <- factor(train.dat[, 'ProductId'], 
                                     levels=product.levels)
  train.dat[, 'UserId'] <- factor(train.dat[, 'UserId'],
                                  levels=user.levels)
  test.dat[, 'ProductId'] <- factor(test.dat[, 'ProductId'], 
                                    levels=product.levels)
  test.dat[, 'UserId'] <- factor(test.dat[, 'UserId'],
                                    levels=user.levels)
  mu <- mean(train.dat[, 'Score'])
  tmp.bU <- train.dat %>% group_by(UserId) %>% summarise(u=mean(Score)) %>% as.data.frame()
  bU <- tmp.bU[, 2]
  names(bU) <- tmp.bU[, 1]
  bU <- bU - mu
  tmp.bV <- train.dat %>% group_by(ProductId) %>% summarise(u=mean(Score)) %>% as.data.frame()
  bV <- tmp.bV[, 2]
  names(bV) <- tmp.bV[, 1]
  bV <- bV - mu
  tmp.train.output <- ComputeLossRMSE(train.dat, U=NA, V=NA, bU, bV, mu, lambda=0, model.type='baseline')
  tmp.test.output <- ComputeLossRMSE(test.dat, U=NA, V=NA, bU, bV, mu, lambda=0, model.type='baseline')
  return(list(train.rmse=tmp.train.output$rmse, test.rmse=tmp.test.output$rmse, bU=bU, bV=bV, mu=mu))
}

# uses stochastic gradient descent to find optimal U and V for a model that 
# only takes into account explicity rating
ComputeSGDRatingOnly <- function(train.dat, test.dat, latent.dim, max.iter=1, 
                                 init.eta=0.02, step.size='fixed', 
                                 decay.pow=0.25, rmse.interval=25, 
                                 model.type='intercept', start.U=NA, 
                                 start.V=NA, start.bU=NA, start.bV=NA, 
                                 start.mu=NA, tol=0.001, lambda=0) {
  # Arguments:
  #   train.dat: training input data, in which each row is a product-user pair and its rating
  #   test.dat: test data
  #   latent.dim: dimension of latent factors
  #   max.iter: maximum number of epochs
  #   init.eta: learning rate
  #   step.size: if 'fixed', then always use init.eta as step size; otherwise, 
  #   use init.eta / iter ^ (decay.pow)
  #   decay.pow: rate of decay for step size if not chosen fixed step size
  #   rmse.interval: number of epochs to compute loss and rmse
  #   model.type: 'intercept': include intercepts; 
  #   otherwise, no intercept, only inner product
  #   start.U: starting value for U
  #   start.V: starting value for V
  #   start.bU: starting value for bU (intercept)
  #   start.bV: starting value for bV (intercept)
  #   start.mu: starting value for intercept
  #   tol: stopping criteria
  #   lambda: weight for L2 regularization for   
  #
  # Returns:
  #   U, V, bU, bV, mu, training rmse, loss, test rmse, loss
  
  # determine the number of products and the number of users
  train.dat <- as.data.frame(train.dat)
  test.dat <- as.data.frame(test.dat)
  # make sure product and user levels are consistent between train set and test set
  # and with bU, bV, U, V names
  product.levels <- unique(train.dat[, 'ProductId'])
  user.levels <- unique(train.dat[, 'UserId'])
  train.dat[, 'ProductId'] <- factor(train.dat[, 'ProductId'], 
                                     levels=product.levels)
  train.dat[, 'UserId'] <- factor(train.dat[, 'UserId'],
                                  levels=user.levels)
  test.dat[, 'ProductId'] <- factor(test.dat[, 'ProductId'], 
                                    levels=product.levels)
  test.dat[, 'UserId'] <- factor(test.dat[, 'UserId'],
                                 levels=user.levels)
  num.train.dat <- nrow(train.dat)
  num.U <- length(user.levels)
  num.V <- length(product.levels)
  
  # initialization
  mean.score <- mean(train.dat[, 'Score'])
  if (is.na(start.U)) {
    U <- matrix(sqrt((mean.score - 1) / latent.dim) + 
                  runif(num.U * latent.dim, min=-0.1, max=0.1),
                nrow=num.U, ncol=latent.dim)
  } else {
    U <- start.U
  }
  rownames(U) <- user.levels
  
  if (is.na(start.V)) {
    V <- matrix(sqrt((mean.score - 1) / latent.dim) + 
                  runif(num.V * latent.dim, min=-0.1, max=0.1),
                nrow=num.V, ncol=latent.dim)
  } else {
    V <- start.V
  }
  rownames(V) <- product.levels
  
  if (is.na(start.bU)) {
    bU <- runif(num.U, min=-0.1, max=0.1)
  } else {
    bU <- start.bU
  }
  names(bU) <- user.levels
  
  if (is.na(start.bV)) {
    bV <- runif(num.V, min=-0.1, max=0.1)
  } else {
    bV <- start.bV
  }
  names(bV) <- product.levels
  
  if (is.na(start.mu)) {
    mu <- runif(1, min=-0.1, max=0.1)
  } else {
    mu <- start.mu
  }
  
  train.rmse <- rep(NA, floor(max.iter/rmse.interval))
  train.loss <- rep(NA, floor(max.iter/rmse.interval))
  test.rmse <- rep(NA, floor(max.iter/rmse.interval))
  test.loss <- rep(NA, floor(max.iter/rmse.interval))
  
  for (iter in 1 : max.iter) {
    if (step.size == 'fixed') {
      eta <- init.eta
    } else {
      eta <- init.eta / iter^(decay.pow)
    }
    #randomly permute the current epoch
    permuted.sample.index <- sample(1 : num.train.dat)
    print(paste0('starting epoch number: ', iter))
    flush.console() 
    pb <- txtProgressBar(min = 0, max = num.train.dat, style = 3)
    i.prog <- 1
    for (r in permuted.sample.index) {
      user.id <- train.dat[r, 'UserId']
      product.id <- train.dat[r, 'ProductId']
      if (model.type == 'intercept') {
        tmp.error <- train.dat[r, 'Score'] - mu - bU[user.id] - bV[product.id] - t(U[user.id, ]) %*% V[product.id, ]
      } else {
        tmp.error <- train.dat[r, 'Score'] - t(U[user.id, ]) %*% V[product.id, ]
      }
    
      U[user.id, ] <- U[user.id, ] + eta * (tmp.error * V[product.id, ] - lambda * U[user.id, ])
      V[product.id, ] <- V[product.id, ] + eta * (tmp.error * U[user.id, ] - lambda * V[product.id, ])
      
      if (model.type == 'intercept') {
        bU[user.id] <- bU[user.id] + eta * (tmp.error - lambda * bU[user.id])
        bV[product.id] <- bV[product.id] + eta * (tmp.error - lambda * bV[product.id])
        mu <- mu + eta * tmp.error
      }
      
      setTxtProgressBar(pb, i.prog)
      i.prog <- i.prog + 1
    }
    close(pb)
    if (iter %% rmse.interval == 0) {
      tmp.index <- floor(iter/rmse.interval)
      print('computing training loss and training rmse')
      flush.console()
      tmp.train.output <- ComputeLossRMSE(train.dat, U, V, bU, bV, mu, lambda, model.type)
      train.rmse[tmp.index] <- tmp.train.output$rmse
      train.loss[tmp.index] <- tmp.train.output$loss
      print('computing test loss and test rmse')
      flush.console()
      tmp.test.output <- ComputeLossRMSE(test.dat, U, V, bU, bV, mu, lambda, model.type)
      test.rmse[tmp.index] <- tmp.test.output$rmse
      test.loss[tmp.index] <- tmp.test.output$loss
      print(paste0('at ', iter, 'th epoch, training loss is ', train.loss[tmp.index], 
                   ' and training rmse is ', train.rmse[tmp.index]))
      print(paste0('at ', iter, 'th epoch, test loss is ', test.loss[tmp.index], 
                   ' and test rmse is ', test.rmse[tmp.index]))
    }
  }
  return(list(train.rmse=train.rmse, train.loss=train.loss, 
              test.rmse=test.rmse, test.loss=test.loss, 
              U=U, V=V, bU=bU, bV=bV, mu=mu))
}

# model comparision through cross-valdiation
CVSelectModel <- function(train.dat, K, max.iter, lambdas, latent.dims, model.types) {
  # split data into K folds
  train.dat <- as.data.frame(train.dat)
  num.dat <- dim(train.dat)[1]
  fold.length <- floor(num.dat / K)
  perm.order.id <- sample(train.dat[, 'Id'])
  fold.out.id <- list()
  fold.in.id <- list()
  Ratio <- vector()
  for (k in 1:K) {
    if (k < K) {
      fold.out.id[[k]] <- perm.order.id[(fold.length*(k-1)+1):(fold.length*k)]
    } else {
      fold.out.id[[k]] <- perm.order.id[(fold.length*(k-1)+1):num.dat]
    }
    tmp.in <- train.dat[!train.dat[, 'Id'] %in% fold.out.id[[k]], ]
    fold.in.id[[k]] <- tmp.in[, 'Id']
    length.in  <- length(as.vector(fold.in.id[[k]]))
    tmp.out <- train.dat[train.dat[, 'Id'] %in% fold.out.id[[k]], ]
    tmp.out <- semi_join(tmp.out, tmp.in, by='ProductId')
    tmp.out <- semi_join(tmp.out, tmp.in, by='UserId')
    fold.out.id[[k]] <- tmp.out[, 'Id']
    length.out  <- length(fold.out.id[[k]])
    Ratio[k] <- length.out / length.in
  }
  num.models <- length(lambdas[])
  output <- data.frame(lambda=lambdas, latent_dim=latent.dims, model_type=model.types, cv_train_rmse=NA, cv_rmse=NA)
  for (m in 1:num.models) {
    print(paste0('model specs lambda: ', lambdas[m], 
                 ' latent dim: ', latent.dims[m], 
                 ' model type: ', model.types[m]))
    flush.console()
    tmp.test.rmse <- 0
    tmp.train.rmse <- 0
    for (k in 1:K) {
      print(paste0('computing fold ', k))
      flush.console()
      cv.train <- train.dat[train.dat[, 'Id'] %in% fold.in.id[[k]], c(2, 3, 7)]
      cv.test <- train.dat[train.dat[, 'Id'] %in% fold.out.id[[k]], c(2, 3, 7)]
      tmp.out <- ComputeSGDRatingOnly(cv.train,
                                      cv.test, 
                                      latent.dim=latent.dims[m], 
                                      max.iter=max.iter, 
                                      init.eta=0.02, step.size='fixed', 
                                      decay.pow=0.25, rmse.interval=max.iter, 
                                      model.type=model.types[m],
                                      lambda=lambdas[m])
      tmp.test.rmse <- tmp.test.rmse + tmp.out$test.rmse[1]
      tmp.train.rmse <- tmp.train.rmse + tmp.out$train.rmse[1]
    }
    output[m, 'cv_rmse'] <- tmp.test.rmse / K
    output[m, 'cv_train_rmse'] <- tmp.train.rmse / K
  }
  return(list(output = output, R = Ratio))
}

