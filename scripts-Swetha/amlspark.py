import os
import sys
import time

os.environ['SPARK_HOME']="/spark/spark-1.6.0-bin-hadoop2.6/"

sys.path.append("/spark/spark-1.6.0-bin-hadoop2.6/python")


try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.mllib.linalg import Vectors
    import numpy as np
    import math
    from itertools import combinations
    import argparse

    def parseByUser(line):
        """
        Parses user line
        :param line:
        :return:
        """
        lineVals = line.split("\t")
        return lineVals[0], (lineVals[1], lineVals[2])

    def parseBySong(line):
        lineVals = line.split("\t")
        return lineVals[1], (lineVals[0], float(lineVals[2]))

    def parseItemline(line):
        """

        :param line: each line is tuple with songid and (userid, rating)
        :return: userid (songid, rating)
        """

        returnElem = list()
        songid = line[0]
        users = line[1]

        for user in users:
            uid = user[0]
            ratid = user[1]
            val1 = uid,(songid,ratid)
            returnElem.append(val1)

        return returnElem


    def getItemPairs(user_id,items_with_rating):
        """
        Get all combination of items per user
        :param user_id:
        :param items_with_rating:
        :return:
        """
        for item1,item2 in combinations(items_with_rating,2):
            return (item1[0],item2[0]),(item1[1],item2[1])

    def calculateSimilarity(item_pair,rating_pairs):
        """
        Calculates dot product of the rating vector
        :param item_pair:
        :param rating_pairs:
        :return:
        """
        dot_xy = 0.0

        for rating_pair in rating_pairs:
            dot_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])

        return item_pair, dot_xy

    def doPreprocess(line):
        """
        Does the pearson preprocessing step, to normalize the ratings

        :param line: #song1 (userid, rating) # song2 (userid, rating)
        :return: song_id, (user_id, rating)
        """
        users_ratings = line[1]
        a = list()
        dctuserratings = {}
        for pair in users_ratings:
            a.append(float(pair[1]))
            dctuserratings[pair[0]] = float(pair[1])
        mn = np.mean(a)
        dn1 = math.sqrt(np.sum((a-mn)**2))

        for item in dctuserratings:
            value = dctuserratings[item]
            if dn1 > 0.0:
                dctuserratings[item] = (value - mn)*1.0/dn1
            else:
                dctuserratings[item] = 0.0
        return line[0], list(dctuserratings.items())

    def parse_argument():
        """
        Code for parsing arguments
        """
        parser = argparse.ArgumentParser(description='Parsing a file.')
        parser.add_argument('--path', nargs=1, required=True)
        args = vars(parser.parse_args())
        return args

    if __name__ == "__main__":
        start_time = time.time()
        master = "local"
        if len(sys.argv) == 2:
            master = sys.argv[1]

        args = parse_argument()
        path = args['path'][0]
        sc = SparkContext(master, "AMLSpark")

        data = sc.textFile(path + "train_0_sub_100k.txt")
        data = data.repartition(8)
        data.count()


        # yy = data.map(lambda x: parseByUser(x))
        #
        # zz1 = yy.map(lambda x: (x[0],[x[1]]))
        # zz2 = zz1.reduceByKey(lambda x,y: x+y)

        # pre-process the data
        item_user_ratings = (data.map(lambda x: parseBySong(x))
              .map(lambda x: (x[0],[x[1]]))
              .reduceByKey(lambda x,y: x+y)
              .map(lambda x: doPreprocess(x)))

        user_items_ratings = (item_user_ratings.flatMap(lambda x: parseItemline(x))
                              .map(lambda x: (x[0],x[1])))
        user_items = user_items_ratings.groupByKey().map(lambda x : (x[0], list(x[1])))

        combination_items = (user_items.filter(lambda x: len(x[1]) > 1)
                          .map(lambda x: getItemPairs(x[0],x[1]))
                          .groupByKey().map(lambda x : (x[0], list(x[1]))))


        similarity_items = (combination_items.map(lambda p: calculateSimilarity(p[0],p[1]))
                            .collect())

        print len(similarity_items)
        print similarity_items
        print("--- %s seconds ---" % (time.time() - start_time))

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)