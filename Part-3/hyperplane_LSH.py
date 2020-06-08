import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC # "Support vector classifier"

class hyperplane_LSH:

    def __init__(self, X, n):
        '''
        param X: feature vectors (numpy array)
        param n: number of hashing fucntions (random hyperplanes)
        '''
        super().__init__()
        self.X = np.copy(X)
        # get max_ and min_ to convert 0-1 hyperplant to our domain :) thus no need to normalise entire dataset
        self.get_max_min()

        self.d = np.shape(X)[1]
        self.n = n
        self.H = []

        # start adding random hyperplanes
        for i in range(n):
            self.H.append(self.get_random_hyperplane())



    def get_random_hyperplane(self):
        ''' 
        use SVM fitted on random data with dimentions d
        '''
        X = []
        Y = [0,1]

        # create a random point (label 0)
        X.append(np.random.uniform(0,1, self.d))

        # select a random point (label 1)
        random_2 = np.random.uniform(0,1, self.d)
        X.append(random_2)

        # move 0-1 X to required values using self.max_ and self.min_
        X = X*(self.max_ - self.min_) + self.min_

        # Make hyperplane (SVC)
        model = SVC(kernel='linear', C=1E10)
        model.fit(X, Y)

        return model

    def get_max_min(self):
        self.max_ = np.max(self.X, axis=0)
        self.min_ = np.min(self.X, axis=0)




def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='r',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



if __name__ == "__main__":
    hyperplane_LSH(np.zeros((10,2)), n=3)