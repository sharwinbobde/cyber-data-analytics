import pandas as pd
import numpy as np
import math

def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

minPrime = 0
maxPrime = int(10e3)
cached_primes = np.array([i for i in range(minPrime,maxPrime) if is_prime(i)])
np.random.shuffle(cached_primes) # shuffle the prime numbers

def simple_hash(integer, primes):
    """Compute a hash value from a list of integers and 3 primes"""
    result = 0
    result = ((result + integer + primes[0]) * primes[1]) % primes[2]
    return result

def tuple_to_int(items):
    # item is a tuple, merge three numbers by appending
    item = 0
    for i in items:
        item += i
        item *= 10e6    # next integer at most 10^6
    return item


# implimentation inspired from 
class count_min:
    def __init__(self, width, d):
        super().__init__()
        self.width = width
        self.d = d
        self.array_ = np.zeros((d, width), dtype=int)

        self.primes = cached_primes[0:d*3].reshape((d,3)) # select primes for each hash function
    
    def add(self, items, count):
        item = tuple_to_int(items)
        for i in range(self.d): # for each function
            hash_val = int(simple_hash(item, self.primes[i]) % self.width)
            self.array_[i,hash_val] += count
    
    def estimate(self, items):
        item = tuple_to_int(items)
        arr = []
        for k in range(self.d): # for every function d get the hash_val and then the count value from array_
            hash_val = int(simple_hash(item, self.primes[k]) % self.width)
            arr.append(self.array_[k,hash_val])
        # return the minimum value
        return np.min(arr)


