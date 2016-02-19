import numpy as np
import matplotlib.pyplot as plt

# distribution by genre
def plotGenreDistribution(items):
    dictItems = {}
    for item in items:
        dictItems[(item[0])] = item[1]

    ind = np.arange(len(dictItems))


    plt.bar(ind, dictItems.values(), width=.5, color='slategrey')
    plt.title("Distribution of songs by genre ", fontsize=16)
    plt.xlabel("Genre")
    plt.ylabel("Song count")
    labs = dictItems.keys()
    plt.xticks(range(len(labs)), labs)
    plt.savefig("song_genre.png")
    plt.show()
    plt.close()
    return

# distribution by rating

def plotRatingDistribution(ratings):
    index = range(1,6,1)
    plt.bar(index, ratings.values(), width=.75, color='slategrey')
    plt.title("Distribution of ratings ", fontsize=16)
    plt.xlabel("Ratings")
    plt.ylabel("Song count (millions)")
    plt.savefig("song_rating.png")
    plt.show()
    plt.close()
    return
#labelpad=35, rotation=0
# top 5 songs
#[(u'72309', 35682), (u'105433', 33954), (u'22763', 33794), (u'123176', 32393), (u'36561', 32074)]

# top 5 genre
#[(u'0', 65499138), (u'134', 5293264), (u'106', 1744753), (u'114', 1295667), (u'54', 567461)]

if __name__ == "__main__":
    genre_distribution =  ([("Rock", 7384), ("Pop", 2955), ("R&B", 2235), ("Country", 1235), ("Rap", 951),
                            (u'Classic Rock', 501), ('Commedy', 409), ("Folk", 259), ("Jazz", 246)])
    plotGenreDistribution(genre_distribution)

    ratings = {u'1': 20.137227, u'3': 11.438138, u'2': 8.356574, u'5': 23.936445, u'4': 12.476243}
    plotRatingDistribution