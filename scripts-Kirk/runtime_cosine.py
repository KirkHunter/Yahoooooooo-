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


    TrainFile1mil  = TrainFile(txt_files[0], 1)
    TrainFile2mil  = TrainFile(txt_files[1], 2)
    TrainFile3mil  = TrainFile(txt_files[2], 3)
    TrainFile4mil  = TrainFile(txt_files[3], 4)
    TrainFile5mil  = TrainFile(txt_files[4], 5)
    TrainFile6mil  = TrainFile(txt_files[5], 6)
    TrainFile7mil  = TrainFile(txt_files[6], 7)
    TrainFile8mil  = TrainFile(txt_files[7], 8)
    TrainFile9mil  = TrainFile(txt_files[8], 9)
    TrainFile10mil = TrainFile(txt_files[9], 10)

    TrainFiles = [TrainFile1mil, TrainFile2mil, TrainFile3mil, TrainFile4mil,
      TrainFile5mil, TrainFile6mil, TrainFile7mil, TrainFile8mil, 
      TrainFile9mil, TrainFile10mil
    ]


    print "\n\nParsing files ... \n\n"

    for TrainFile in TrainFiles:
        TrainFile.read_lines()
        TrainFile.parse_train_lines()




    # get combinations
    print "\n\nGrouping by users ... \n\n"

    for TrainFile in TrainFiles:
        TrainFile.get_users()



    # isntantiate values dictionary
    vals = defaultdict(dict)


    print "\n\nGetting user counts per file ... \n\n"

    for TrainFile in TrainFiles:
        vals[TrainFile.n]["users"] = TrainFile.get_users_count()




    ###############################################################################
    # Computing cosine similarities
    
    for TrainFile in TrainFiles:

        print "\n\nGetting item combinations %dmil rows... \n\n" % (TrainFile.n)

        TrainFile.item_combos()       

    


    print "\n\nComputing similarities ... \n\n"

    for TrainFile in TrainFiles:
        TrainFile.compute_cosine_similarities()




    for TrainFile in TrainFiles:

        print "\n\nCollecting %dmil dictionary ... \n\n" % (TrainFile.n)

        TrainFile.print_iteration_time(vals, sim='cosine')

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
    plt.savefig('runtime_mil.png')




