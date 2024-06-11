import multiprocessing
from multiprocessing import Process
import sys
import random
from random import randrange
import time


def helloworld(threadnum):
    seconds = randrange(1, 5)
    time.sleep(seconds)
    print(f"Hello from thread {threadnum}")


if __name__ == '__main__':
    processes = []
    for i in range(int(sys.argv[1])):
        process = Process(target=helloworld, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
