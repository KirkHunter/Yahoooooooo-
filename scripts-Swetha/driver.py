
import os
import sys
import lsh
from pyspark import SparkContext
from pyspark import SparkConf
import time
from pyspark.mllib.linalg import SparseVector
import functools

import scipy.spatial.distance as distance
import argparse
import numpy as np

def parseBySong(line):
    #"userid<TAB>songid<TAB>rating"
    lineVals = line.split("\t")
    return lineVals[1], lineVals[0]


def read_song_text(sc, path, num):
    row_data = sc.textFile(path)
    row_data = row_data.repartition(100)
    item_user_ratings = row_data.map(lambda x: parseBySong(x))
    item_user_ratings2 = item_user_ratings.groupByKey()
    item_list = item_user_ratings2.mapValues(lambda x : sorted(map(int, list(x))))
    item_list1 = item_list.map(lambda x : (x[0],SparseVector(num, list(x[1]), np.ones(len(x[1])))))
    rdd_zip_item_id = item_list1.zipWithIndex()
    return rdd_zip_item_id

if __name__ == "__main__":
    start_time = time.time()
    file_name = "file_1_mil.txt"
    sc = SparkContext(conf = SparkConf())
    d = {}

    d[file_name] = 10000

    AWS_ACCESS_KEY_ID = "awskey"
    AWS_SECRET_ACCESS_KEY = "awsaccesskey"

    sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)
    for item in d:
        name = item

        fp = "s3n://bucket-data/"+name
        p = d[item]
        data = read_song_text(sc, fp, p)
        # sparse vector from data
        data_sparse_list = data.map(lambda x:(x[0][1], x[1]))
        rdd_1 = data.map(lambda x:(x[0][0], x[1]))

        # initialize the parameters
        m, n, b, c = 1000,1000,25,2
        # create the model for LSH
        model = lsh.run(data_sparse_list, p, m, n, b, c)

        print ("start printing filename %s" % name)
        # Get similar buckets
        cnt = model.buckets.count()
        # print result time taken
        timetaken = (time.time() - start_time)
        print 'Found %s clusters.' % cnt
        print("--- %s seconds ---" % timetaken)

        tup = cnt, timetaken
        d[name]  = cnt , (time.time() - start_time)

    # write log file
    f = open('output.txt', 'a')
    mystr = str(time.time())

    for item in d:
        mystr1 = str(item) + str(d[item])  +"\n"
        mystr += mystr1

    f.write(mystr)
    f.close()
