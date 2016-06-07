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

    train_file100  = train_file(txt_files[0], 100)
    train_file200  = train_file(txt_files[1], 200)
    train_file300  = train_file(txt_files[2], 300)
    train_file400  = train_file(txt_files[3], 400)
    train_file500  = train_file(txt_files[4], 500)
    train_file600  = train_file(txt_files[5], 600)
    train_file700  = train_file(txt_files[6], 700)
    train_file800  = train_file(txt_files[7], 800)
    train_file900  = train_file(txt_files[8], 900)
    train_file1mil = train_file(txt_files[9], 1000)

    train_files = [train_file100, train_file200, train_file300, train_file400,
      train_file500, train_file600, train_file700, train_file800, 
      train_file900, train_file1mil
    ]


    print "\n\nParsing files ... \n\n"

    for train_file in train_files:
        train_file.read_lines()
        train_file.parse_train_lines()




    # get combinations
    print "\n\nGrouping by users ... \n\n"

    for train_file in train_files:
        train_file.get_users()



    # isntantiate values dictionary
    vals = defaultdict(dict)


    print "\n\nGetting user counts per file ... \n\n"

    for train_file in train_files:
        vals[train_file.n]["users"] = train_file.get_users_count()




    ###############################################################################
    # Computing cosine similarities
    
    for train_file in train_files:
        if train_file.n < 1000:
            print "\n\nGetting item combinations %dk rows... \n\n" % (train_file.n)
        else:
            print "\n\nGetting item combinations 1mil rows... \n\n"

        train_file.item_combos()       

    


    print "\n\nComputing similarities ... \n\n"

    for train_file in train_files:
        train_file.compute_cosine_similarities()




    for train_file in train_files:

        if train_file.n < 1000:
            print "\n\nCollecting %dk dictionary ... \n\n" % (train_file.n)
        else:
            print "\n\nCollecting 1mil dictionary ... \n\n"

        train_file.print_iteration_time(vals, sim='cosine')

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




