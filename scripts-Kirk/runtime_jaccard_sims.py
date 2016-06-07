from collections import defaultdict
from function_library import *
from itertools import combinations
from pyspark import SparkContext
from random import sample
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time



sc = SparkContext()


sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)




if __name__ == '__main__':


    train_file = train_file('s3n://sparkstuff/train_0_sub_30mil.txt', 30)

    train_file.read_lines()
    train_file.parse_train_lines()


	print "\n\nCreating song count dictionaries ... \n\n"
	train_file.get_songs()
    train_file.create_songs_map()



	# get user counts
	print "\n\nGrouping by users ... \n\n"
	train_file.get_users()


	vals = defaultdict(dict)
	print "\n\nGetting user counts per file ... \n\n"
	vals[30]["users"] = train_file.get_users_count()



	print "\n\nGetting item combinations 30mil rows... \n\n"
	train_file.item_combos()


	print "\n\nComputing similarities ... \n\n"
	train_file.compute_jaccard_similarities()


	print "\n\nCollecting 30mil dictionary ... \n\n"
	start = time.time()
	train_file.create_songs_map()
	vals[30]["time"] = time.time()-start

	print "\n\n"
	for kv in sorted(vals.iteritems()):
	    print kv
	print "\n\n"





