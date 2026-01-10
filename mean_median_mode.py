import random
import statistics

def mean_median_mode(num):
    return statistics.mean(num), statistics.median(num), statistics.mode(num)

num = [random.randint(100, 150)  for i in range(100)]

mean , median , mode = mean_median_mode(num)

print("mean : ", mean)
print("median : ", median)
print("mode:", mode)
