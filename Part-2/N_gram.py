import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from sklearn.neighbors import KNeighborsClassifier

class N_gram_Component:

    '''
    deep[] = list for frequency tables
    deep[i] = frequency table for ith feature
    deep[i][sym] = frequency for ith feature

    Notes:
    default frequency = 1 to apply Laplacean smoothing
    frequency tables stored as dictionaries so I can input keys as "3|,1,2" (which is 3 given 1 and 2 occured)
    '''

    def __init__(self, N, n_features):
        super().__init__()
        self.N = N
        self.deep = []
        self.n_features = n_features
        self.build_default_N_gram

    def build_default_N_gram(self):
        # insert default values
        for i in range(self.n_features):
            self.deep.append(defaultdict(lambda :1)) # Laplacean Smoothing 
        return
    
    def get_N_gram_signature(self, X):
        self.build_default_N_gram() # clean the slate

        # populate it
        for k in range(np.shape(X)[0] - self.N): # move through time
            for j in range(self.n_features): # one-feature at a time
                sym = '' 
                given = ''
                for i in range(self.N): # build symbol as we progress
                    sym = str(X[k+i,j])
                    if i == 0: # unigram
                        key = sym
                    else:
                        key = sym + '|' + given
                    given += "," + sym # inconsistent : gives c|,a,b but works fine :) 
                self.deep[j][key] += 1
                # print(str(self.deep[j][key])+": " +key) # example 26: 95|,96,96,96,96
        
        # now generate a signature
        signature = []
        self.bins = np.unique(X)
        combinations = list(itertools.combinations_with_replacement(self.bins, self.N-1))
        givens=[]
        for comb in combinations:
            given = '|'
            for sym in comb:
                given += ',' + str(sym)
            givens.append(given)

        for j in range(self.n_features): # one-feature at a time
                for a in self.bins:
                    for given in givens:
                        key = str(a)+given
                        # print(key)
                        signature.append(self.deep[j][key])
        return signature

    @staticmethod
    def discretise(X, window_size, levels=3, overlap=0):
        '''
        Input:
            X = numpy array
            window_size = number of samples in 1 window
            overlap = number of samples that overlap in two windows. Can be -ve to show jumping/skipping samples
        '''
        if overlap > window_size:
            print("keep overlap < window_size")
            exit(0)

        # normalise
        min_ = np.min(X, axis=0)
        max_ = np.max(X, axis=0)
        diff = (max_ - min_)
        diff[diff == 0] = 1 # diff for constant signals can become 0
        X_disc = np.apply_along_axis(lambda row : (row - min_)/diff * 100, axis=1, arr=X)

        # convert to percentile levels
        bins = range(0, 100, int(100/levels))
        X_disc = np.apply_along_axis(lambda row : np.digitize(row, bins).astype(int), axis=1, arr=X_disc)

        X_windowed = np.empty(shape=(0, np.shape(X)[1]),dtype=np.int)
        i = 0;
        while i + window_size < np.shape(X_disc)[0]:
            j = i + window_size
            X_windowed = np.append(X_windowed, [np.mean(X_disc[i:j, :], axis=0).astype(int)], axis=0)
            i = j - overlap
        return X_windowed

    
    @staticmethod
    def discretise_labels(Y, window_size, overlap=0):
        if overlap > window_size:
            print("keep overlap < window_size")
            exit(0)
        disc_labels = []

        i = 0;
        while i + window_size < np.shape(Y)[0]:
            j = i + window_size
            disc_labels.append(Y[j-1])
            i = j - overlap

        return np.array(disc_labels)


    def generate_profiles(self, X, L:int):
        # iterate through X making windows of size L and find all signatures
        i = 0
        profiles = []
        while i+L < np.shape(X)[0]:
            sig = self.get_N_gram_signature(X[i:i+L, :])
            profiles.append(sig)   
            i=i+L
        return np.array(profiles)
    
    def generate_profile_labels(self, Y, L:int):
        # Y is all labels for discretised signal
        i = 0
        profile_labels = []
        while i+L < np.shape(Y)[0]:
            profile_labels.append(Y[i+L-1])   
            i=i+L
        profile_labels = np.array(profile_labels)
        profile_labels = profile_labels.reshape((profile_labels.shape[0], 1))
        return profile_labels

    def fit_kNN(self, X, Y):
        # Norm 2 Eucledean for k-NN the same as Cosine similarity ranking and classification
        # k arbitrarily picked because assignment goal is NOT to optimise k-NN
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean', algorithm='ball_tree', n_jobs=-1) 
        self.knn.fit(X,Y)
        return


    def classify(self, X, L):
        return self.knn.predict(X)


if __name__ == "__main__":
    data_1 = pd.read_csv("./Part-2/data/BATADAL_dataset03.csv", delimiter=r",\s{0,1}")
    data_1["DATETIME"] = pd.to_datetime(data_1.DATETIME)
    signals = ['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
       'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3', 'F_PU4', 'S_PU4',
       'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6', 'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8',
       'F_PU9', 'S_PU9', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2',
       'S_V2', 'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
       'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
    X = data_1[signals].to_numpy()
    labels = data_1['ATT_FLAG'].to_numpy()
    X_labels = np.zeros(np.shape(X)[0])
    print(labels.shape)

    L=50

    N_gram = N_gram_Component(N=5, n_features=43)
    X_windowed = N_gram.discretise(X, window_size=5, levels=10,  overlap=2)

    print("making profiles")
    X_profiles = N_gram.generate_profiles(X_windowed, L)
    print(X_profiles.shape)
    X_profile_labels = N_gram.generate_profile_labels(X_labels, L)