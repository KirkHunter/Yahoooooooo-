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
# train_raw100 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_100k.txt')
# train_raw150 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_150k.txt')
# train_raw200 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_200k.txt')
# train_raw250 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_250k.txt')
# train_raw300 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_300k.txt')
# train_raw350 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_350k.txt')
# train_raw400 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_400k.txt')
# train_raw450 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_450k.txt')
# train_raw500 = sc.textFile('/Users/kirkhunter/Documents/MSAN/630 Advanced ML/project/data/train/train_0_sub_500k.txt')


# we read in files of various sizes to compare runtime

train_raw100 = sc.textFile('s3n://sparkstuff/train_0_sub_100k.txt')
train_raw100 = train_raw100.repartition(50)

train_raw200 = sc.textFile('s3n://sparkstuff/train_0_sub_200k.txt')
train_raw200 = train_raw200.repartition(50)

train_raw300 = sc.textFile('s3n://sparkstuff/train_0_sub_300k.txt')
train_raw300 = train_raw300.repartition(50)

train_raw400 = sc.textFile('s3n://sparkstuff/train_0_sub_400k.txt')
train_raw400 = train_raw400.repartition(50)

train_raw500 = sc.textFile('s3n://sparkstuff/train_0_sub_500k.txt')
train_raw500 = train_raw500.repartition(50)

train_raw600 = sc.textFile('s3n://sparkstuff/train_0_sub_600k.txt')
train_raw600 = train_raw600.repartition(50)

train_raw700 = sc.textFile('s3n://sparkstuff/train_0_sub_700k.txt')
train_raw700 = train_raw700.repartition(50)

train_raw800 = sc.textFile('s3n://sparkstuff/train_0_sub_800k.txt')
train_raw800 = train_raw800.repartition(50)

train_raw900 = sc.textFile('s3n://sparkstuff/train_0_sub_900k.txt')
train_raw900 = train_raw900.repartition(50)

train_raw1mil = sc.textFile('s3n://sparkstuff/train_0_sub_1mil.txt')
train_raw1mil = train_raw1mil.repartition(50)


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
    train100 = train_raw100.map(parse_line).cache()
    train200 = train_raw200.map(parse_line).cache() 
    train300 = train_raw300.map(parse_line).cache() 
    train400 = train_raw400.map(parse_line).cache() 
    train500 = train_raw500.map(parse_line).cache() 
    train600 = train_raw600.map(parse_line).cache()
    train700 = train_raw700.map(parse_line).cache()
    train800 = train_raw800.map(parse_line).cache() 
    train900 = train_raw900.map(parse_line).cache() 
    train1mil = train_raw1mil.map(parse_line).cache() 


    # get combinations
    print "\n\nGrouping by users ... \n\n"
    users100 = train100.groupByKey()
    users200 = train200.groupByKey()
    users300 = train300.groupByKey()
    users400 = train400.groupByKey()
    users500 = train500.groupByKey() 
    users600 = train600.groupByKey()
    users700 = train700.groupByKey()
    users800 = train800.groupByKey()
    users900 = train900.groupByKey() 
    users1mil = train1mil.groupByKey()

    vals = defaultdict(dict)
    print "\n\nGetting user counts per file ... \n\n"
    vals[100]["users"] = users100.count()
    vals[200]["users"] = users200.count()
    vals[300]["users"] = users300.count()
    vals[400]["users"] = users400.count()
    vals[500]["users"] = users500.count()
    vals[600]["users"] = users600.count()
    vals[700]["users"] = users700.count()
    vals[800]["users"] = users800.count()
    vals[900]["users"] = users900.count()
    vals[1000]["users"] = users1mil.count()



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

    print "\n\nGetting item combinations 100k rows... \n\n"
    item_combos100 = (users100.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 200k rows... \n\n"
    item_combos200 = (users200.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 300k rows... \n\n"
    item_combos300 = (users300.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 400k rows... \n\n"
    item_combos400 = (users400.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 500k rows... \n\n"
    item_combos500 = (users500.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 600k rows... \n\n"
    item_combos600 = (users600.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 700k rows... \n\n"
    item_combos700 = (users700.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 800k rows... \n\n"
    item_combos800 = (users800.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 900k rows... \n\n"
    item_combos900 = (users900.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()

    print "\n\nGetting item combinations 1mil rows... \n\n"
    item_combos1mil = (users1mil.map(lambda line: interaction_cut(line, p))
                         .filter(lambda line: len(line[1]) > 1)
                         .flatMap(get_song_combinations)
                         .groupByKey()
                         .mapValues(list) ).cache()


    print "\n\nComputing similarities ... \n\n"
    cosine_sims100 = item_combos100.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims200 = item_combos200.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims300 = item_combos300.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims400 = item_combos400.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims500 = item_combos500.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims600 = item_combos600.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims700 = item_combos700.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims800 = item_combos800.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims900 = item_combos900.map(lambda x: cosine_sim(x[0], x[1])).cache()
    cosine_sims1mil = item_combos1mil.map(lambda x: cosine_sim(x[0], x[1])).cache()


    print "\n\nCollecting 100k dictionary ... \n\n"
    start = time.time()
    # cosine_sims100_dict = cosine_sims100.collectAsMap()
    cosine_sims100_dict = cosine_sims100.count()
    vals[100]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 200k dictionary ... \n\n"
    start = time.time()
    # cosine_sims200_dict = cosine_sims200.collectAsMap()
    cosine_sims200_dict = cosine_sims200.count()
    vals[200]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 300k dictionary ... \n\n"
    start = time.time()
    # cosine_sims300_dict = cosine_sims300.collectAsMap()
    cosine_sims300_dict = cosine_sims300.count()
    vals[300]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 400k dictionary ... \n\n"
    start = time.time()
    # cosine_sims400_dict = cosine_sims400.collectAsMap()
    cosine_sims400_dict = cosine_sims400.count()
    vals[400]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 500k dictionary ... \n\n"
    start = time.time()
    # cosine_sims500_dict = cosine_sims500.collectAsMap()
    cosine_sims500_dict = cosine_sims500.count()
    vals[500]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 600k dictionary ... \n\n"
    start = time.time()
    # cosine_sims600_dict = cosine_sims600.collectAsMap()
    cosine_sims600_dict = cosine_sims600.count()
    vals[600]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 700k dictionary ... \n\n"
    start = time.time()
    # cosine_sims700_dict = cosine_sims700.collectAsMap()
    cosine_sims700_dict = cosine_sims700.count()
    vals[700]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 800k dictionary ... \n\n"
    start = time.time()
    # cosine_sims800_dict = cosine_sims800.collectAsMap()
    cosine_sims800_dict = cosine_sims800.count()
    vals[800]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 900k dictionary ... \n\n"
    start = time.time()
    # cosine_sims900_dict = cosine_sims900.collectAsMap()
    cosine_sims900_dict = cosine_sims900.count()
    vals[900]["time"] = time.time()-start

    print "\n\n"
    for kv in sorted(vals.iteritems()):
        print kv
    print "\n\n"

    print "\n\nCollecting 1mil dictionary ... \n\n"
    start = time.time()
    # cosine_sims1mil_dict = cosine_sims1mil.collectAsMap()
    cosine_sims1mil_dict = cosine_sims1mil.count()
    vals[1000]["time"] = time.time()-start

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
plt.savefig('runtime.png')




