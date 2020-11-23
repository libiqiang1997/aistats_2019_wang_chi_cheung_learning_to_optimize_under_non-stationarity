from datetime import datetime
import multiprocessing as mp
import numpy as np

if __name__ == '__main__':
    T = 2.4 * 10 ** 5
    print('T:', T)
    ss = [3, 1] + [0.1 * (i + 1) for i in range(5)]
    for s in ss:
        length = round(T / 5 * s, 2)
        print(s)
        print(length)
        counnts = T / length
        print(counnts)
