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


    txt_files = ['s3n://sparkstuff/train_0_sub_' + str(n) + 'mil.txt' for n in range(1, 11)]

    
    train_file1mil  = train_file(txt_files[0], 1)
    train_file2mil  = train_file(txt_files[1], 2)
    train_file3mil  = train_file(txt_files[2], 3)
    train_file4mil  = train_file(txt_files[3], 4)
    train_file5mil  = train_file(txt_files[4], 5)
    train_file6mil  = train_file(txt_files[5], 6)
    train_file7mil  = train_file(txt_files[6], 7)
    train_file8mil  = train_file(txt_files[7], 8)
    train_file9mil  = train_file(txt_files[8], 9)
    train_file10mil = train_file(txt_files[9], 10)

    train_files = [train_file1mil, train_file2mil, train_file3mil, train_file4mil,
      train_file5mil, train_file6mil, train_file7mil, train_file8mil, 
      train_file9mil, train_file10mil
    ]


    print "\n\nParsing files ... \n\n"

    for train_file in train_files:
        train_file.read_lines()
        train_file.parse_train_lines()




    # get combinations
    print "\n\nCreating song count dictionaries ... \n\n"

    for train_file in train_files:
        train_file.get_songs()
        train_file.create_songs_map()




    # get user counts
    print "\n\nGrouping by users ... \n\n"

    for train_file in train_files:
        train_file.get_users()




    # isntantiate values dictionary
    vals = defaultdict(dict)

    print "\n\nGetting user counts per file ... \n\n"

    for train_file in train_files:
        vals[train_file.n]["users"] = train_file.get_users_count()






    ###############################################################################
    # # Computing Jaccard similarities
    
    for train_file in train_files:

        print "\n\nGetting item combinations %dmil rows... \n\n" % (train_file.n)

        train_file.item_combos()       

    

    print "\n\nComputing similarities ... \n\n"

    for train_file in train_files:
        train_file.create_songs_map()
        train_file.compute_jaccard_similarities(train_file.songs_dict)





    for train_file in train_files:

        print "\n\nCollecting %dmil dictionary ... \n\n" % (train_file.n)

        train_file.print_iteration_time(vals, sim='jaccard')

        print "\n\n"
        for kv in sorted(vals.iteritems()):
            print kv
        print "\n\n"


    

    ###########################################################################
    # plot the results

    users, runtime = [], []
    for k, v in sorted(vals.iteritems()):
        users.append(v["users"])
        runtime.append(v["time"])

    fig, ax = plt.subplots()
    plt.plot(users, runtime, "bo", users, runtime, "k")
    plt.title('Runtime Increase for Growing User Base', size="x-large")
    plt.xlabel('$N-$ users (thousands)', labelpad=5, size="large")
    plt.ylabel('time (seconds)', labelpad=55, size="large", rotation=0)
    plt.savefig('runtime_mil_jaccard.png')




