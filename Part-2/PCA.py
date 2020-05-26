from sklearn.decomposition import PCA
import numpy as np

class PCA_Component:
    def __init__(self):
        super().__init__()
    
    def set_X(self, X):
        self.X = X
        self.max_ = np.max(X)
        self.min_ = np.min(X)
        X_normed = (X - self.min_)/(self.max_ - self.min_) # normalise
        self.X_normed = X_normed

    def PCA_fit_transform(self, n_components):
        # new PCA started whenever we calculate a new encoding
        self.pca = PCA(n_components=n_components)
        encoding = self.pca.fit_transform(self.X_normed)

        # keep residual max min correct
        R = self.calculate_residual(self.X)
        R = np.sum(R, axis=1) # sum up residuals for all signals
        self.max_R = np.max(R)
        self.min_R= np.min(R)

        return encoding

    def PCA_transform(self, Z):
        Z_normed = (Z - self.min_)/(self.max_ - self.min_) # normalise
        encoding = self.pca.fit_transform(Z_normed)
        return encoding
    
    def calculate_residual(self, Z):
        encoding = self.PCA_transform(Z)
        Z_normed_inv = self.pca.inverse_transform(encoding)

        Z_normed = (Z - self.min_)/(self.max_ - self.min_) # normalise
        return (Z_normed - Z_normed_inv)**2     # Return squared error

    def anomaly_score(self, Z):
        R = self.calculate_residual(Z)
        R = np.sum(R, axis=1) # sum up residuals for all signals
        # use the best min and max
        max_ = np.max([np.max(R), self.max_R]) 
        min_ = np.min([np.min(R), self.min_R])
        scores = (R - min_)/(max_ - min_)
        return scores

    def classify(self, X):
        threshold = 0.03

        scores = self.anomaly_score(X)
        bins = [0.0,threshold]
        pred = np.digitize(scores, bins) -1
        return pred