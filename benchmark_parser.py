import matplotlib.pyplot as plt
import re
import numpy as np

ITERS              = 20
MAX_THREADS        = 28
FILE_START_PADDING = 7
filename           = "d2q9-bgk.out"

def read_file(filename="d2q9-bgk.out", padding=7, iters=20, max_threads=28):
    averages = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for threads in range(1, max_threads):
            padding += 1 # Account for the printing of num of threads
            av = 0
            for i in range(iters):
                exe_line = lines[padding + ((threads - 1) * max_threads) + i + 5]        
                exe_time = float(re.findall(r"/([+-]?([0-9]*[.])?[0-9]*)+/g", exe_line).group())
                av += exe_time
            averages.append(av / iters)
    return averages

def graph_averages(averages, max_threads=28):
    x = np.arange(1, max_threads)
    plt.bar(x, averages)
    plt.show()

