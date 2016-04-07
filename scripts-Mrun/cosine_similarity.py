__author__ = 'mrunmayee'


# This file calculates the cosine similarity between all possible item-item pairs using MapReduce technique in Spark.

from pyspark import SparkContext
import numpy as np

# AWS keys
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""

sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)


def split_data(x):
    """
    This function returns the user item rating as a key value pair with user as the key and item and rating as a
    tuple value.
    :param x: User rating item record
    :return: A key value pair
    """
    sp_x = x.split(",")
    return int(sp_x[0]), (int(sp_x[1]), float(sp_x[2]))

def group_func(ls):
    """
    This function creates key value pairs of all possible item pairs as the key and a tuple of their respective
    ratings as the value.
    :param ls: A list of item rating tuple for a user
    :return: A list of key value pair
    """
    ls_tuples = []
    for i in ls:
        for j in ls:
            if j[0] > i[0]:
                ls_tuples.append(((i[0], j[0]), (i[1] * j[1])))
    return ls_tuples

def sqr_item(ls):
    """
    This function squares the rating for all items of a user.
    :param ls: A list of tuples of item and rating
    :return: A list of tuples of item and squared rating
    """
    ls_tuples = []
    for i in ls:
        sqr = i[1] ** 2
        ls_tuples.append((i[0], sqr))
    return ls_tuples

def cosine_sim(tup, dict_den):
    """
    This function calculates the cosine similarity between two items.
    :param tup: A tuple of items
    :param dict_den: A dictionary with item and squared rating as key value pair
    :return: Similarity between a pair of items
    """
    sim = round((tup[1] * 1.0) / (dict_den[tup[0][0]] * dict_den[tup[0][1]]), 2)
    return sim

# Files on AWS
# data = sc.wholeTextFiles("s3n://..../*.txt").repartition(16)
# data = sc.textFile("s3n://..../train_0.txt").repartition(16)

# Files on local machine
data = sc.textFile("/Users/mrunmayee/Downloads/Data/ydata-ymusic-user-song-ratings-meta-v1_0 2/train_0.txt").repartition(32)
# data = sc.textFile("/Users/mrunmayee/AMLProject/Data/train_0_sub_1mil.txt").repartition(32)


# The input file is in the format user, item, rating
sp_data = data.map(lambda x: split_data(x))
# Convert the item rating tuple for a user into a list
red_sp_data = sp_data.mapValues(lambda x: [x]).reduceByKey(lambda x, y: x + y)
# Create item item pairs for all users
red = red_sp_data.flatMap(lambda x: group_func(x[1]))
# Add the ratings of same item pairs
num_rdd = red.reduceByKey(lambda x, y: x + y)
# Square the rating for each item for each user and store it in a dictionary to pass it to 'cosine_sim' function
squares = red_sp_data.flatMap(lambda x: sqr_item(x[1]))
den_rdd = squares.reduceByKey(lambda x, y: x + y).map(lambda x: (x[0], round(np.sqrt(x[1]), 2)))
dict_den = {}
for i in den_rdd.collect():
    dict_den[i[0]] = i[1]
# Calculate cosine similarity and create an item by item similarity matrix
ibyi = num_rdd.map(lambda x: (x[0], cosine_sim(x, dict_den)))