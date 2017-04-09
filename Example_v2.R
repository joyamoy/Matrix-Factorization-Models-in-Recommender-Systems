## a small simulated dataset testing the matrix factorization technique in collaborative filtering


example <- read.csv("example.csv")

num.dat <- nrow(example)
num.U <- length(unique(example[, 'User']))
num.V <- length(unique(example[, 'Item']))
R.bar <- mean(example[, 'Rating'])
latent.dim <- 1

ComputeLossRMSE <- function(dat, U, V, alpha, beta, mu, lambda.U=0, lambda.V=0) {
  sse <- 0
   num.dat <- nrow(dat)
  for (r in 1 : num.dat) {
    user <- dat[r, 'User']
    item <- dat[r, 'Item']
    tmp.score <- dat[r, 'Rating']
    pred.score <- t(U[user, ]) %*% V[item, ] + alpha[user] + beta[item] + mu
    print(paste0('user is', user, 'item is', item))
    print(paste0('true score is', tmp.score))
    print(paste0('pred score is', pred.score))
    flush.console()
    sse <- sse + (tmp.score - pred.score)^2
  }
  rmse <- sqrt(sse / nrow(dat))
  loss <- sse + lambda.U * (sum(U^2) + sum(alpha^2)) + lambda.V * (sum(V^2) + sum(beta^2))
  return(list(rmse=rmse, loss=loss))
}

example[,'User'] <- factor(example[,'User'], levels=unique(example[,'User']))
example[,'Item'] <- factor(example[,'Item'], levels=unique(example[,'Item']))

#initialize parameters 
U <- matrix(sqrt((R.bar) / latent.dim) + 
              runif(num.U * latent.dim, min=-0.1, max=0.1),
            nrow=num.U, ncol=latent.dim)
rownames(U) <- unique(example[, 'User'])

V <- matrix(sqrt((R.bar) / latent.dim) + 
              runif(num.V * latent.dim, min=-0.1, max=0.1),
            nrow=num.V, ncol=latent.dim)
rownames(V) <- unique(example[, 'Item'])

alpha <- runif(num.U, min=-0.1, max=0.1)
names(alpha) <- unique(example[, 'User'])
beta <- runif(num.V, min=-0.1, max=0.1)
names(beta) <- unique(example[, 'Item'])
mu <- runif(1, min=-0.1, max=0.1)


##Fixed Step size
eta <- 0.07
lambda.U <- 0.001
lambda.V <- 0.001
max.iter <- 5
rmse <- rep(NA, max.iter)
loss <- rep(NA, max.iter)
mu.all <- list()
alpha.all <- list()
beta.all <- list()
U.all <- list()
V.all <- list()
each.permuted.index <- matrix(NA, nrow=max.iter, ncol=num.dat)


for (iter in 1:max.iter){
  print(paste0('iteration', iter))
  flush.console()
  permuted.sample.index <- sample(1 : num.dat)
  for (r in permuted.sample.index) {
    user <- example[r, 'User']
    item <- example[r, 'Item']
    tmp.error <- example[r, 'Rating'] - mu - alpha[user] - beta[item] - t(U[user, ]) %*% V[item, ]
    
    U[user, ] <- U[user, ] + eta * (tmp.error * V[item, ] - lambda.U * U[user, ])
    V[item, ] <- V[item, ] + eta * (tmp.error * U[user, ] - lambda.V * V[item, ])
    alpha[user] <- alpha[user] + eta * (tmp.error - lambda.U * alpha[user])
    beta[item] <- beta[item] + eta * (tmp.error - lambda.V * beta[item])
    mu <- mu + eta * tmp.error
  }
  mu.all[[iter]] <- mu
  alpha.all[[iter]] <- alpha
  beta.all[[iter]] <- beta
  U.all[[iter]] <- U
  V.all[[iter]] <- V
  tmp.output <- ComputeLossRMSE(example, U, V, alpha, beta, mu, lambda.U, lambda.V)
  rmse[iter] <- tmp.output$rmse
  loss[iter] <- tmp.output$loss
  each.permuted.index[iter, ] <- permuted.sample.index
}
  
#examine parameters and results
rmse
loss
each.permuted.index
mu
alpha
beta
U
V
mu.all
alpha.all
beta.all
U.all
V.all

# make predictions for the complete user-item matrix
pred.ratings <- as.data.frame(matrix(NA, nrow=num.U, ncol=num.V))
colnames(pred.ratings) <- unique(example[, 'Item'])
rownames(pred.ratings) <- unique(example[, 'User'])

for (u in 1:num.U){
  for (v in 1:num.V){
    user <- rownames(pred.ratings)[u]
    item <- colnames(pred.ratings)[v]
    pred.ratings[user, item] <- mu + alpha[user] + beta[item] + t(U[user, ]) %*% V[item, ]
  }
}

# original user-item matrix (input)
obs.ratings <- as.data.frame(matrix(NA, nrow=num.U, ncol=num.V))
colnames(obs.ratings) <- unique(example[, 'Item'])
rownames(obs.ratings) <- unique(example[, 'User'])

obs.ratings[1, 1] <- 3
obs.ratings[2, 1] <- 4
obs.ratings[2, 2] <- 2
obs.ratings[3, 3] <- 5
obs.ratings[3, 4] <- 4


pred.ratings
obs.ratings
