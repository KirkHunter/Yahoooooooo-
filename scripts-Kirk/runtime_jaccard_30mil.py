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


train_raw30mil = sc.textFile('s3n://sparkstuff/train_0_sub_30mil.txt')
train_raw30mil = train_raw30mil.repartition(50)


def parse_line(line):
    user, song, rating = line.split('\t')
    return (user, (song, float(rating)))

def get_song_and_rating(line):
    user, song, rating = line[0], line[1][0], line[1][1]
    return song, (float(rating))

def pearson(song1, song2, ratings, song_averages_dict):
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
    return line[0], (line[1][0], line[1][1])

def interaction_cut(line, p):
    user, items = line[0], line[1]
    if len(items) > p:
        return (user, sample(items, p))
    return (user, items)

def songs_ratings_list(pair1, pair2):
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


	train30mil = train_raw30mil.map(parse_line)

	print "\n\nCreating song count dictionaries ... \n\n"
	songs30mil = (train30mil.map(key_by_song_jaccard)
	                 .reduceByKey(lambda x,y: x+y)
	                 .collectAsMap() )



	# get user counts
	print "\n\nGrouping by users ... \n\n"
	users30mil = train30mil.groupByKey().cache()


	vals = defaultdict(dict)
	print "\n\nGetting user counts per file ... \n\n"
	vals[30]["users"] = users30mil.count()


	p = 15

	print "\n\nGetting item combinations 30mil rows... \n\n"
	item_combos30mil = (users30mil.map(lambda line: interaction_cut(line, p))
	                    .filter(lambda line: len(line[1]) > 1)
	                    .flatMap(get_song_combinations_jaccard)
	                    .reduceByKey(lambda x,y: x+y) )


	print "\n\nComputing similarities ... \n\n"
	jaccard_sims30mil = item_combos30mil.map(
	                    lambda x: jaccard(x[0][0], x[0][1], x[1], songs30mil))


	print "\n\nCollecting 30mil dictionary ... \n\n"
	start = time.time()
	jaccard_sims30mil_dict = jaccard_sims30mil.count()
	vals[30]["time"] = time.time()-start

	print "\n\n"
	for kv in sorted(vals.iteritems()):
	    print kv
	print "\n\n"





