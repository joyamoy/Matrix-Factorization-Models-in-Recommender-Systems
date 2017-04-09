## Apply K-means clustering based on the extracted item feature vectors using matrix factorization model without intercept

nointercept_V_0.05_35 <- read.csv("U:/2_3_2017_kmeans_clustering/nointercept_V_0.05_35.csv")


ProductId <- as.factor(rownames(nointercept_V_0.05_35))
V <- as.matrix(nointercept_V_0.05_35)

km.out <- kmeans(V,5,nstart=20)
km.out$cluster


table(km.out$cluster)

library(readr)
library(dplyr)
km.result <- data.frame((km.out$cluster))
temp.name.km <- rownames(km.result)
km.result <- data.frame(cbind(temp.name.km,km.result))
names(km.result) <- c('ProductId','cluster')
write_csv(km.result, 'km.result.csv')


head(km.result)
head(train.dat)
##Combine Reviews_train with km.result by ProductId
train.dat <- read_csv("Reviews_train.csv")
train.set.cluster <- full_join(train.dat, km.result, by='ProductId')
head(train.set.cluster)
write_csv(train.set.cluster,"km.result.combine.csv")


aggregate(train.set.cluster$Score, list(train.set.cluster$cluster), mean)

cluster_1 <- train.set.cluster[train.set.cluster$cluster==1 , ]
write_csv(cluster_1, 'cluster_1.csv')
cluster_2 <- train.set.cluster[train.set.cluster$cluster==2 , ]
write_csv(cluster_2, 'cluster_2.csv')
cluster_3 <- train.set.cluster[train.set.cluster$cluster==3 , ]
write_csv(cluster_3, 'cluster_3.csv')
cluster_4 <- train.set.cluster[train.set.cluster$cluster==4 , ]
write_csv(cluster_4, 'cluster_4.csv')
cluster_5 <- train.set.cluster[train.set.cluster$cluster==5 , ]
write_csv(cluster_5, 'cluster_5.csv')


### Use Elbow Method to determine number of clusters k
nointercept_V_0.05_35 <- read.csv("U:/2_3_2017_kmeans_clustering/nointercept_V_0.05_35.csv")
ProductId <- as.factor(rownames(nointercept_V_0.05_35))
V <- as.matrix(nointercept_V_0.05_35)



WCSS <- c(0)
for (k in 1:20){
  km.out <- kmeans(V, k ,nstart=20)
  WCSS <- rbind(WCSS, km.out$tot.withinss)
}


plot(1:20, WCSS[2:21],pch=19,
     ylab="WCSS(k)",xlab="number of clusters k",  main = "Model without Intercept", cex.lab=1.5)

plot(1:15, WCSS[2: 16],pch=19,
     ylab="WCSS(k)",xlab="number of clusters k", main = "Model with Intercept", cex.lab=1.5)


write.table(WCSS[2:21], './WCSS_nointercept.csv', row.names = FALSE, col.names = FALSE, sep = ',')

