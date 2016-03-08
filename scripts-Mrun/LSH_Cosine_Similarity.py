__author__ = 'mrunmayee'

import os
import sys

os.environ['SPARK_HOME']="/Users/mrunmayee/Downloads/spark-1.6.0-bin-hadoop2.6/"

sys.path.append("/Users/mrunmayee/Downloads/spark-1.6.0-bin-hadoop2.6/python/")

# Append pyspark  to Python Path
sys.path.append("/Users/zjb238/code/spark/python")
sys.path.append("/Users/zjb238/code/spark/python/lib/py4j-0.9-src.zip")
import py4j
from pyspark import SparkContext
from pyspark import SparkConf

sc = SparkContext()

import numpy as np
import scipy.sparse as sps
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Matrices
from random import gauss
import math
import time

start_time = time.time()

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""

sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

path = "s3n://macproj/train_1m.txt"
print "\n\n\n", path, "\n\n\n"

np.random.seed(10)
count_users = 0


# Original data item : (user, rating)
def split_data(x):
    sp_x = x.split("\t")
    return int(sp_x[1]), (int(sp_x[0]), float(sp_x[2]))


# Convert to a sparse vector format
def conv(x, y):
    return x[0] + y[0], x[1] + y[1]


# Convert to a sparse vector format by sorting the indices
def sort_index(x):
    x = list(x)
    m = sorted(x[0])
    n = [x[1] for (x[0],x[1]) in sorted(zip(x[0],x[1]))]
    return m, n


# Create sparse vectors 'sp_vecs' is an RDD of sparse vectors
def conv_sv(x):
    return Vectors.sparse(count_users, x[1][0], x[1][1])


# This function creates random vectors with a norm = 1 and dimension = no. of users
def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def find_hash12(x, list_rand_vecs):
    # global ito, it
    ones_zeroes = np.zeros(no_vectors)
    #it += 1
    # ito += 1
    for i in xrange(0, no_vectors):
        if list_rand_vecs[i].dot(x) >= 0:
            ones_zeroes[i] = 1.0
        # else: ones_zeroes.append(0)

    # return (it, ones_zeroes)
    return ones_zeroes

def filter_func(x):
    if x[0][0] < x[1][0]:
        return x


def group_func(x):
    return ((x[0][0], x[1][0]), hamming(x[0][1], x[1][1]))


# Hamming distance gives the probability that the two items are not similar
def hamming(x,y):
    """Calculate the Hamming distance between two vectors"""
    assert len(x) == len(y)
    c = 0
    for i in xrange(0, len(x)):
        if x[i] != y[i]:
            c += 1
    return c * 1.0/ len(x)


def calc_cosine(x):
    return (x[0], round(math.cos(x[1] * math.pi), 4))


data = sc.textFile(path).repartition(32)

or_data = data.map(lambda x: split_data(x))

# Convert values to list
sp_data = or_data.map(lambda x: ((x[0]), ([x[1][0]], [x[1][1]])))

sp_format = sp_data.reduceByKey(lambda x, y: conv(x, y))


cn = sp_format.flatMap(lambda x: x[1][0]).distinct()

cv = cn.collect()
count_users = len(cv)
sorted_indices = sorted(cv)

sv = sp_format.mapValues(lambda x: sort_index(x))

sp_vecs = sv.map(lambda x: conv_sv(x))

# Create a list of vectors
list_rand_vecs = []
no_vectors = 50
for i in xrange(0, no_vectors):
    list_rand_vecs.append(Vectors.sparse(count_users, sorted_indices, make_rand_vector(count_users)))


hashed = sp_vecs.map(lambda x: find_hash12(x, list_rand_vecs))
m = hashed.zipWithIndex().map(lambda x: (x[1] + 1, x[0]))

n = m.cartesian(m)

filt = n.filter(lambda x: filter_func(x))
ls_ham = filt.map(lambda x: group_func(x))
ls_cosine = ls_ham.map(lambda x: calc_cosine(x))

print "\n\n\n\n\nTotal_time", time.time() - start_time, "sec\n\n\n\n\n"
print "\n\n\n\n\n", ls_cosine.first(), "\n\n\n\n\n"