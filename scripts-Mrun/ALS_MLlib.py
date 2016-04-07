__author__ = 'mrunmayee'

# This code uses the ALS algorithm from the Spark MLlib library to predict recommendations.

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import numpy as np
# Path for training file on AWS
# data = sc.textFile("s3n://sparkstuff/train_0_sub_30mil.txt").repartition(16)

# Local path for training file
data = sc.textFile("/Users/mrunmayee/AMLProject/Data/train_0_sub_1mil.txt")

# Split the data set into training and test data set
train_set, test_set = data.map(lambda l: l.split('\t')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).randomSplit([8, 2], 10)

# Set the rank and iterations arguments for the ALS algorithm
rank = 10
num_iterations = 10

# Train the model with the training data set
model = ALS.train(train_set, rank, num_iterations)

testdata = test_set.map(lambda p: (p[0], p[1]))

# Using the model, predict the ratings for the test data set
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

# Join the preeicted and original ratings in the test data set using user-item as the key
ratesAndPreds = test_set.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

# Calculate the
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
RMSE = np.sqrt(MSE)
print("Root Mean Squared Error = " + str(RMSE))