from sklearn.decomposition import PCA
import numpy as np

class PCA_Component:
    def __init__(self):
        super().__init__()
    
    def set_X(self, X):
        self.X = X
        max_ = np.max(X)
        min_ = np.min(X)
        X_normed = (X - min_)/(max_ - min_) # normalise
        self.X_normed = X_normed

    def PCA_fit_transform(self, n_components):
        # new PCA started whenever we calculate a new encoding
        self.pca = PCA(n_components=n_components)
        encoding = self.pca.fit_transform(self.X_normed)
        return encoding

    def PCA_transform(self, Z):
        max_ = np.max(Z)
        min_ = np.min(Z)
        Z_normed = (Z - min_)/(max_ - min_) # normalise
        encoding = self.pca.fit_transform(Z_normed)
        return encoding
    
    def calculate_residual(self, Z):
        max_ = np.max(Z)
        min_ = np.min(Z)
        Z_normed = (Z - min_)/(max_ - min_) # normalise

        encoding = self.PCA_transform(Z)
        Z_normed_inv = self.pca.inverse_transform(encoding)

        return (Z_normed - Z_normed_inv)**2     # Return squared error

    def anomaly_score(self, Z,):
        R = self.calculate_residual(Z)
        R = np.sum(R, axis=1) # sum up residuals for all signals
        max_ = np.max(R)
        min_ = np.min(R)
        scores = (R - min_)/(max_ - min_)
        return scores