from pyspark import SparkContext
import numpy as np
from collections import defaultdict
from random import sample
from itertools import combinations
import time
import matplotlib
import matplotlib.pyplot as plt

sc = SparkContext()


sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)


# train_raw50 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_50k.txt')
# train_raw1mil = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_1milk.txt')
# train_raw150 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_150k.txt')
# train_raw2mil = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_2milk.txt')
# train_raw250 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_250k.txt')
# train_raw3mil = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_3milk.txt')
# train_raw350 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_350k.txt')
# train_raw4mil = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_4milk.txt')
# train_raw450 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_450k.txt')
# train_raw5mil = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_5milk.txt')


train_raw1mil = sc.textFile('s3n://sparkstuff/train_0_sub_1mil.txt')
train_raw1mil = train_raw1mil.repartition(50)

train_raw2mil = sc.textFile('s3n://sparkstuff/train_0_sub_2mil.txt')
train_raw2mil = train_raw2mil.repartition(50)

train_raw3mil = sc.textFile('s3n://sparkstuff/train_0_sub_3mil.txt')
train_raw3mil = train_raw3mil.repartition(50)

train_raw4mil = sc.textFile('s3n://sparkstuff/train_0_sub_4mil.txt')
train_raw4mil = train_raw4mil.repartition(50)

train_raw5mil = sc.textFile('s3n://sparkstuff/train_0_sub_5mil.txt')
train_raw5mil = train_raw5mil.repartition(50)

train_raw6mil = sc.textFile('s3n://sparkstuff/train_0_sub_6mil.txt')
train_raw6mil = train_raw6mil.repartition(50)

train_raw7mil = sc.textFile('s3n://sparkstuff/train_0_sub_7mil.txt')
train_raw7mil = train_raw7mil.repartition(50)

train_raw8mil = sc.textFile('s3n://sparkstuff/train_0_sub_8mil.txt')
train_raw8mil = train_raw8mil.repartition(50)

train_raw9mil = sc.textFile('s3n://sparkstuff/train_0_sub_9mil.txt')
train_raw9mil = train_raw9mil.repartition(50)

train_raw10mil = sc.textFile('s3n://sparkstuff/train_0_sub_10mil.txt')
train_raw10mil = train_raw10mil.repartition(50)


def parse_line(line):
    '''parse tab delimited data'''
    user, song, rating = line.split('\t')
    return (user, (song, float(rating)))

def get_song_and_rating(line):
    '''key by song to get average ratings'''
    user, song, rating = line[0], line[1][0], line[1][1]
    return song, (float(rating))

def pearson(song1, song2, ratings, song_averages_dict):
    '''calculate similarity using pearson correlation
       takes a list of ratings for a pair of songs
         [(r11, r21), (r12, r22), ...]
        and takes the sum of the difference between each rating and the 
        song average, as well as the L2 norm of each song's ratings. 
    '''
    s1_avg = song_averages_dict[song1]
    s2_avg = song_averages_dict[song2]
    num = 0.
    diff1 = 0.
    diff2 = 0.
    for r1, r2 in ratings:
        num += (r1 - s1_avg) * (r2 - s2_avg)
        diff1 += (r1 - s1_avg)**2
        diff2 += (r2 - s2_avg)**2
    den = np.sqrt(diff1 * diff2)
    return ((song1, song2), num / den if den else 0)

def cosine_sim(pair, ratings):
    '''calculate similarity using cosine similarity
       takes a list of ratings for a pair of songs
        [(r11, r21), (r12, r22), ...]
        and takes the sum of the product between each song rating
        as well as the L2 norm of each song's ratings
    '''
    diff1 = 0.
    diff2 = 0.
    numerator = 0.
    n = 0
    for r1, r2 in ratings:
        diff1 += r1**2
        diff2 += r2**2
        numerator += r1 * r2
        n += 1
    denominator = np.sqrt(diff1 * diff2)
    cosine = numerator / denominator if denominator else 0.
    return pair, (cosine, n)

def jaccard(song1, song2, dot, songs_dict):
    '''given two songs, and their dot product (number of users who have
        rated both song1 and song2), returns the jaccard similarity
        measure for the pair. Given by: 

          dot / (n_i + n_j - dot)

        where dot is as defined above, ni equals number of users who 
        have rated song i, and nj equals number of users who have rated 
        song j.   
    '''
    sim = float(dot) / (songs_dict[song1] + songs_dict[song2] - dot)
    return (song1, song2), sim

def key_by_song_jaccard(line):
    '''key by song to obtain number of users who have
       rated song i for all i
    '''
    song = line[1][0]
    return song, 1

def key_by_user(line):
    '''takes a line with format (user, song, rating)
       returns (user, (song, rating))
    '''
    return line[0], (line[1][0], line[1][1])

def interaction_cut(line, p):
    '''perform the interaction cut, 
       if a user has given more than p ratings,
       then replace those ratings with a sample of p 
       of those ratings
    '''
    user, items = line[0], line[1]
    if len(items) > p:
        return (user, sample(items, p))
    return (user, items)

def songs_ratings_list(pair1, pair2):
    '''given two tuples with (song, rating), 
       return a tuple ( (song1, song2), (rating1, rating2) )
       where the key is the (song1, song2),
       where song1 < song2 to avoid duplicate key pairs
    '''
    song1, rating1 = pair1[0], pair1[1]
    song2, rating2 = pair2[0], pair2[1]
    if song1 < song2:
        return (song1, song2), (rating1, rating2)
    else:
        return (song2, song1), (rating2, rating1)

def songs_ratings_list_jaccard(pair1, pair2):
    '''given two pairs of (song, rating), return the pair of songs
       as a key, this allows for a quick reduceByKey operation
       to obtain the dot product for song_i and song_j for all 
       i and j.
    '''
    song1 = pair1[0]
    song2 = pair2[0] 
    if song1 < song2:
        return (song1, song2), (1)
    else:
        return (song2, song1), (1)

def get_song_combinations(line):
    '''given a list of user ratings, return a list of all combinations 
       of pairs of songs and their respective ratings.
        returns (si, sj), (ri, rj) for all i not equal to j and j > i. 
    '''
    songs = line[1]
    combos = combinations(songs, 2)
    return [songs_ratings_list(p1, p2) for p1, p2 in combos]

def get_song_combinations_jaccard(line):
    '''use the song_ratings_list_jaccard function to 
       get song combinations.
    '''
    songs = line[1]
    combos = combinations(songs, 2)
    return [songs_ratings_list_jaccard(p1, p2) for p1, p2 in combos]


if __name__ == '__main__':

    print "\n\nParsing files ... \n\n" 
    train1mil = train_raw1mil.map(parse_line)
    train2mil = train_raw2mil.map(parse_line)
    train3mil = train_raw3mil.map(parse_line)
    train4mil = train_raw4mil.map(parse_line)
    train5mil = train_raw5mil.map(parse_line)
    train6mil = train_raw6mil.map(parse_line)
    train7mil = train_raw7mil.map(parse_line)
    train8mil = train_raw8mil.map(parse_line)
    train9mil = train_raw9mil.map(parse_line)
    train10mil = train_raw10mil.map(parse_line)


    # get combinations
    print "\n\nCreating song count dictionaries ... \n\n"
    songs1mil = (train1mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs2mil = (train2mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs3mil = (train3mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs4mil = (train4mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs5mil = (train5mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs6mil = (train6mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs7mil = (train7mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs8mil = (train8mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs9mil = (train9mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    songs10mil = (train10mil.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y)
                     .collectAsMap() )

    # get user counts
    print "\n\nGrouping by users ... \n\n"
    users1mil = train1mil.groupByKey().cache()
    users2mil = train2mil.groupByKey().cache()
    users3mil = train3mil.groupByKey().cache()
    users4mil = train4mil.groupByKey().cache()
    users5mil = train5mil.groupByKey().cache()
    users6mil = train6mil.groupByKey().cache()
    users7mil = train7mil.groupByKey().cache()
    users8mil = train8mil.groupByKey().cache()
    users9mil = train9mil.groupByKey().cache() 
    users10mil = train10mil.groupByKey().cache()

    vals = defaultdict(dict)
    print "\n\nGetting user counts per file ... \n\n"
    vals[1]["users"] = users1mil.count()
    vals[2]["users"] = users2mil.count()
    vals[3]["users"] = users3mil.count()
    vals[4]["users"] = users4mil.count()
    vals[5]["users"] = users5mil.count()
    vals[6]["users"] = users6mil.count()
    vals[7]["users"] = users7mil.count()
    vals[8]["users"] = users8mil.count()
    vals[9]["users"] = users9mil.count()
    vals[10]["users"] = users10mil.count()



    ###############################################################################
    # # Computing Pearson similarities

    # create averages dict

    # averages_dict = train.map(get_song_and_rating).groupByKey().mapValues(list).mapValues(
    #     lambda x: np.mean(x)).collectAsMap()

    # len(averages_dict.keys())

    # users_cut = users.map(lambda line: interaction_cut(line, 300))

    # item_combos = (users_cut.filter(lambda line: len(line[1]) > 1)
    #                    .flatMap(get_song_combinations)
    #                    .groupByKey()
    #                    .mapValues(list) )

    # item_sims = item_combos.map(lambda x: pearson(x[0][0], x[0][1], x[1], averages_dict))

    ###############################################################################





    ###############################################################################
    # # Computing cosine similarities
    p = 15

    print "\n\nGetting item combinations 1mil rows... \n\n"
    item_combos1mil = (users1mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 2mil rows... \n\n"
    item_combos2mil = (users2mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 3mil rows... \n\n"
    item_combos3mil = (users3mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 4mil rows... \n\n"
    item_combos4mil = (users4mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 5mil rows... \n\n"
    item_combos5mil = (users5mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 6mil rows... \n\n"
    item_combos6mil = (users6mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 7mil rows... \n\n"
    item_combos7mil = (users7mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 8mil rows... \n\n"
    item_combos8mil = (users8mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 9mil rows... \n\n"
    item_combos9mil = (users9mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )

    print "\n\nGetting item combinations 10mil rows... \n\n"
    item_combos10mil = (users10mil.map(lambda line: interaction_cut(line, p))
                        .filter(lambda line: len(line[1]) > 1)
                        .flatMap(get_song_combinations_jaccard)
                        .reduceByKey(lambda x,y: x+y) )


    print "\n\nComputing similarities ... \n\n"
    jaccard_sims1mil = item_combos1mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs1mil))

    jaccard_sims2mil = item_combos2mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs2mil))

    jaccard_sims3mil = item_combos3mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs3mil))

    jaccard_sims4mil = item_combos4mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs4mil))

    jaccard_sims5mil = item_combos5mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs5mil))

    jaccard_sims6mil = item_combos6mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs6mil))

    jaccard_sims7mil = item_combos7mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs7mil))

    jaccard_sims8mil = item_combos8mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs8mil))

    jaccard_sims9mil = item_combos9mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs9mil))

    jaccard_sims10mil = item_combos10mil.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs10mil))



    print "\n\nCollecting 1mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims1mil_dict = jaccard_sims1mil.count()
    vals[1]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 2mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims2mil_dict = jaccard_sims2mil.count()
    vals[2]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 3mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims3mil_dict = jaccard_sims3mil.count()
    vals[3]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 4mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims4mil_dict = jaccard_sims4mil.count()
    vals[4]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 5mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims5mil_dict = jaccard_sims5mil.count()
    vals[5]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 6mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims6mil_dict = jaccard_sims6mil.count()
    vals[6]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 7mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims7mil_dict = jaccard_sims7mil.count()
    vals[7]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 8mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims8mil_dict = jaccard_sims8mil.count()
    vals[8]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 9mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims9mil_dict = jaccard_sims9mil.count()
    vals[9]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 10mil dictionary ... \n\n"
    start = time.time()
    jaccard_sims10mil_dict = jaccard_sims10mil.count()
    vals[10]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"


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




