__author__ = 'mrunmayee'

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1 = [12.96, 22.7, 31.88, 46.3, 53.9, 64.21, 75.57, 98.67, 102.38, 114.02]
y = [round(i / 60.0, 2) for i in y1]
print y


x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y11 = [6.86, 11.60, 16.54, 22.36, 27.44, 33.10, 38.47, 45.26, 52.4, 58.59]
y2 = [round(i / 60.0, 2) for i in y11]
print y2
plt.figure(1)
# plt.subplot(122)
plt.axis([0.5, 11.5, 0, 2])
plt.plot(x2, y, 'r-', linewidth = 1.0)
plt.plot(x2, y2, 'b-', linewidth = 1.0)
plt.xlabel('Ratings (millions)', fontsize=15, color='black')
plt.ylabel('Time (hours)', fontsize=15, color='black')
plt.title('Runtime vs Number of Ratings (LSH - Cosine Similarity)', fontsize=16)
plt.text(9, 0.75, "Vectors = 50", size = 14)
plt.text(9, 1.6, "Vectors = 100", size = 14)


plt.show()
