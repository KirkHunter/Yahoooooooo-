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


    txt_files = ['s3n://sparkstuff/train_0_sub_' + str(n) + 'k.txt' for n in range(100, 1000)]
    txt_files += ['s3n://sparkstuff/train_0_sub_1mil.txt']

    TrainFile100  = TrainFile(txt_files[0], 100)
    TrainFile200  = TrainFile(txt_files[1], 200)
    TrainFile300  = TrainFile(txt_files[2], 300)
    TrainFile400  = TrainFile(txt_files[3], 400)
    TrainFile500  = TrainFile(txt_files[4], 500)
    TrainFile600  = TrainFile(txt_files[5], 600)
    TrainFile700  = TrainFile(txt_files[6], 700)
    TrainFile800  = TrainFile(txt_files[7], 800)
    TrainFile900  = TrainFile(txt_files[8], 900)
    TrainFile1mil = TrainFile(txt_files[9], 1000)

    TrainFiles = [TrainFile100, TrainFile200, TrainFile300, TrainFile400,
      TrainFile500, TrainFile600, TrainFile700, TrainFile800, 
      TrainFile900, TrainFile1mil
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
        if TrainFile.n < 1000:
            print "\n\nGetting item combinations %dk rows... \n\n" % (TrainFile.n)
        else:
            print "\n\nGetting item combinations 1mil rows... \n\n"

        TrainFile.item_combos()       

    


    print "\n\nComputing similarities ... \n\n"

    for TrainFile in TrainFiles:
        TrainFile.compute_cosine_similarities()




    for TrainFile in TrainFiles:

        if TrainFile.n < 1000:
            print "\n\nCollecting %dk dictionary ... \n\n" % (TrainFile.n)
        else:
            print "\n\nCollecting 1mil dictionary ... \n\n"

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
    plt.savefig('runtime.png')




