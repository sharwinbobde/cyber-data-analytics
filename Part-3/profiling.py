import pandas as pd
from nltk import ngrams
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from hyperplane_LSH import generate_signature
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


def profile_scenario(filename):
    df = pd.read_csv(filename)
    n = 2

    groupby_cols = ['SrcAddr' ] 
    groupby_cols = df[groupby_cols].to_numpy()
    groupby_cols_unique = np.unique(groupby_cols, axis=0)

    # know all possible signatures
    unique_ngrams = np.unique(np.array(list(ngrams(df['TotBytes_Dur'].to_numpy(), n))), axis=0)

    ip_ngrams = defaultdict(list)

    # get signatures (counts of ngrams)
    for group in tqdm(groupby_cols_unique):
        is_in_group =  groupby_cols[:,0] == group[0]
        indexes = np.where(is_in_group)
        items = df['TotBytes_Dur'].to_numpy()[indexes]

        # do sliding window
        L = 50 # window length
        i = 0
        increment = 25
        while i + L < items.shape[0]:
            # make ngrams
            temp_grams = list(ngrams(items[i: i+L], n))
            i += increment # increment counter.. not used afterwords
            if np.shape(temp_grams)[0] == 0: # has < n entries
                continue
            # make ngram frequency profile
            signature = generate_signature(temp_grams, unique_ngrams)
            # append
            ip_ngrams[group[0]].append(signature)

    return ip_ngrams


# def compute_cosine_distance(ip_ngrams:dict, train_IP:str, infedted_IPs:list, all_IPs:list):
#     A = []  # set A: signatures for 1 infected IP 
#     for signature in ip_ngrams[train_IP]:
#         A.append(signature)

#     true_labels = [] # labels for signatures 
#     ip_distance_mean_std = {}
#     for ip in all_IPs:
#         if ip in infedted_IPs:
#             true_labels.append[1]
#         else:
#             true_labels.append[0]

#         distances = []
#         B = [] # set B: signatures for other IP's
#         for signature in ip_ngrams[ip]:
#             for a in A:
#                 distances.append(cosine_distances(a, signature))
#         ip_distance_mean_std[ip] = [np.mean(distances), np.std(distances)]

#     return ip_distance_mean_std, true_labels
    