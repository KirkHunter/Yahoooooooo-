from collections import defaultdict
from itertools import combinations
from pyspark import SparkContext
from random import sample
import numpy as np
import time




class TrainFile(object):
    def __init__(self, filename, n):
        self.filename = filename
        self.n = n

    def read_lines(self):
        lines = sc.textFile(filename)
        lines.repartition(50)
        self.lines = lines

    def parse_train_lines(self):
        lines = self.lines
        lines = lines.map(parse_line).cache()
        self.parsed_lines = lines

    def get_users(self):
        parsed_lines = self.parsed_lines
        self.users = parsed_lines.groupByKey()

    def get_songs(self):
        parsed_lines = self.parsed_lines
        self.songs = (parsed_lines.map(key_by_song_jaccard)
                     .reduceByKey(lambda x,y: x+y) )

    def get_songs_count(self):
        return self.songs.count()

    def create_songs_map(self):
        self.songs_dict = self.songs.collectAsMap()

    def get_users_count(self):
        return self.users.count()

    def item_combos(self, p=15):
        users = self.users
        combos = (users.map(lambda line: interaction_cut(line, p))
                            .filter(lambda line: len(line[1]) > 1)
                            .flatMap(get_song_combinations)
                            .groupByKey()
                            .mapValues(list) ).cache()
        self.item_combos = combos

    def compute_cosine_similarities(self):
        combos = self.item_combos
        cosine_sims = combos.map(lambda x: cosine_sim(x[0], x[1])).cache()
        self.cosine_sims = cosine_sims

    def compute_jaccard_similarities(self, songs_dict):
        jaccard_sims = self.item_combos.map(
                        lambda x: jaccard(x[0][0], x[0][1], x[1], songs_dict))
        self.jaccard_sims = jaccard_sims

    def print_iteration_time(self, vals_dict, sim):
        start = time.time()
        
        if sim == 'cosine':
            sims_dict = self.cosine_sims.count()
        if sim == 'jaccard':
            sims_dict = self.jaccard_sims.count()
        else:
            sims_dict = {}

        vals_dict[self.n]["time"] = time.time()-start








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

