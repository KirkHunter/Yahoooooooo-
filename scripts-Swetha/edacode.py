

import os
import sys
import time


os.environ['SPARK_HOME']="/spark/spark-1.6.0-bin-hadoop2.6/"

sys.path.append("/spark/spark-1.6.0-bin-hadoop2.6/python")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    import pyspark.mllib.linalg.distributed.CoordinateMatrix
    import pyspark.mllib.linalg.distributed.MatrixEntry
    from pyspark.mllib.linalg import Vectors
    import numpy as np

    print ("Successfully imported Spark Modules")


    if __name__ == "__main__":
        start_time = time.time()
        master = "local"
        sc = SparkContext(master, "WordCount")

        path = "/ydata-ymusic-user-song-ratings-meta-v1_03/"
        data = sc.textFile(path + "train_0_sub_100k.txt")

        data = data.repartition(8)
        data.count()

        # distribution of the ratings
        drt = data.map(lambda x: (x.split("\t")[2], 1))
        drt1 = drt.reduceByKey(lambda x,y: x+y).collectAsMap()

        # distribution of songs by genre
        #  "song id<TAB>albumid<TAB>artist id<TAB>genre id"
        gendata = sc.textFile(path+"song-attributes.txt")
        gdata = gendata.map(lambda x: (x.split("\t")[0], x.split("\t")[3]))


        gendata1 = (gendata.map(lambda x: (x.split("\t")[3], 1))
            .reduceByKey(lambda x,y: x+y)
            .takeOrdered(10, key = lambda x: -x[1])
                    )

        ###Top 5 genre
        data = sc.textFile(path + "train_0.txt")
        data = data.repartition(8)
        songs_ratings = data.map(lambda x: (x.split("\t")[1], 1))
        song_attributes = (sc.textFile(path+"song-attributes.txt")
                           .map(lambda x: (x.split("\t")[0], x.split("\t")[3])))

        top5genre = (song_attributes.join(songs_ratings)
                     .map(lambda x: (x[1][0], x[1][1]))
                     .reduceByKey(lambda x,y: x+y)
                     .takeOrdered(5, key= lambda x:-x[1]))

        ###Top 5 songs
        top5songs = (songs_ratings.reduceByKey(lambda x,y: x+y)
                     .takeOrdered(5, key = lambda x: -x[1]))
except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


