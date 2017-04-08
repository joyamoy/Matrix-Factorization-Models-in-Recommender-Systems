# split the full dataset into training and test sets
require(readr)
require(dplyr)

set.seed(1688)

# read in full dataset
amazon.reviews.full.dat <- read_csv("Reviews.csv")

# decide train:test ratio
train.test.ratio <- 0.9
test.prop <- 1 / (train.test.ratio + 1)

# randomly sample rows from the full dataset
test.set <- sample_frac(amazon.reviews.full.dat, size=test.prop, replace=FALSE)

# training set is the full dataset leaving out the test set
train.set <- anti_join(amazon.reviews.full.dat, test.set, by='Id')

test.set <- semi_join(test.set, train.set, by='ProductId')
test.set <- semi_join(test.set, train.set, by='UserId')

dim(train.set)/dim(test.set)


write_csv(train.set, "./Reviews_train.csv")
write_csv(test.set, "./Reviews_test.csv")
