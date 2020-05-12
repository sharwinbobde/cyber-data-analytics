import numpy as np
import math


class SMOTE:
    """
    Class for doing Synthetic Minority Oversampling Technique
    """

    def __init__(self, p: float, k: int, random_state: int = 1337) -> None:
        """
        Parameters:
        p: Percentage of the minority class required after oversampling

        k: Number of nearest neighbours to consider while generating samples

        random_state: Random seed

        """

        # initialize variables
        self.p = p
        self.k = k
        self.nn = None
        self.X_min = None
        self.X_maj = None
        self.y = None
        self.minority_label = None

        # set random seed
        np.random.seed(random_state)

    def euclidean_distance(self, v1: np.array, v2: np.array) -> np.array:
        """
        Computes Euclidean distance between
        corresponding rows in `v1` and `v2`.

        Parameters:
        v1: NxM numpy array with each row as a sample

        v2: NxM numpy array with each row as a sample

        Returns:
        Nx1 numpy array with each row being the euclidean distance between
        corresponding rows of `v1` and `v2`.
        """
        # compute euclidean distance
        return np.sqrt(np.sum(np.square(v1 - v2), axis=1))

    def get_nearest_neighbours(
        self, sample: np.array, population: np.array
    ) -> np.array:
        """
        Computes `k` nearest-neighbours of `sample`
        present in array of vectors `population`

        Parameters:
        sample: 1xM numpy array representing a single sample of data

        population: NxM numpy array with all samples in the population

        Returns:
        sort indices of neighbours of `sample` in `population`
        """
        # create copies of `sample` to compare to
        # every other sample in the population
        sample_duplicates = np.tile(sample, (population.shape[0], 1))

        # compute euclidean distances
        distances = self.euclidean_distance(population, sample_duplicates)

        # return the indices used to sort the samples
        # according to euclidean distance
        return np.argsort(distances)

    def get_minority_label(self, labels: np.array) -> (int, np.array):
        """
        Get the label which is the minority in terms of frequency

        Parameters:
        labels: Nx1 numpy array of labels

        Returns:
        minority_label: label with lowest frequency in `labels`
        minority_label_map: boolean array indicating indices corresponding
                            to minority labels
        """
        # get the counts of each distinct label
        counts = np.bincount(labels)
        label = np.nonzero(counts)[0]
        label_counts = list(zip(label, counts[label]))

        # sort the label counts in ascending order
        label_counts.sort(key=lambda x: x[1])

        # get the minority class labels
        minority_label = label_counts[0][0]

        # get the boolean map where label is the minority label
        minority_label_map = labels == minority_label
        return minority_label, minority_label_map

    def get_synthetic_sample(self, sample: np.array, neighbours: np.array) -> np.array:
        """
        Return a synthetic sample according to the SMOTE algorithm

        Parameters:
        sample: 1xM sample of data
        neighbours: NxN sort indices of neighbours of `sample`.

        Returns:
        synthetic_sample: 1xM synthetic sample according to SMOTE
        """
        # pick a random nearest neighbour index
        nn_index = np.random.randint(0, high=self.k)

        # pick a sample from minority samples using index from above
        # choose from 1 to k+1 since 0th nearest neighbour is the sample itself
        nearest_neighbours = self.X_min[neighbours][1 : self.k + 1]
        nn_sample = nearest_neighbours[nn_index]

        # choose a random weight for the neighbour
        weight = np.random.uniform(low=0, high=1)

        # generate synthetic sample by weighting sample and random neighbour
        synthetic_sample = sample + (sample - nn_sample) * weight
        return synthetic_sample

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Get the nearest neighbours of the data

        Parameters:
        X: NxM dataset with each row containing a sample
        y: Nx1 labels
        """

        # get the minority label
        # and the boolean map for minority samples
        minority_label, minority_label_map = self.get_minority_label(y)

        # use the boolean map to choose the minority samples
        X_min = X[minority_label_map, :]

        # since with this SMOTE, we would only like to do oversampling,
        # if desired percentage for minority class < current ratio
        # raise an exception
        if self.p <= 100 * X_min.shape[0] / X.shape[0]:
            raise ValueError(
                f"""minority class in X already has a percentage of {round(100*X_min.shape[0]/X.shape[0], 2)} which is >= desired percentage self.p = {self.p}. This class is used to do oversampling of minority class, not undersampling"""
            )

        # get the sort indices for nearest neighbours of
        # each sample in the minority class
        self.nn = np.apply_along_axis(self.get_nearest_neighbours, 1, X_min, X_min)

        # set variables as class variables
        self.minority_label = minority_label
        self.y = y
        self.X_min = X_min

        # select majority class samples using the boolean map
        self.X_maj = X[~minority_label_map, :]

    def transform(self, shuffle: bool = True) -> (np.array, np.array):
        """
        Generate the samples according to the nearest neighbours
        computed in `self.fit` and the desired minority class percentage
        in `self.p`.

        Parameters:
        shuffle: boolean parameter indicating whether
                 final dataset is to be shuffled

        Returns:
        X_resampled: minority oversampled dataset

        y_resampled: labels corresponding to the oversampled dataset
        """
        num_maj_samples = self.X_maj.shape[0]
        num_min_samples = self.X_min.shape[0]

        # self.p = 100 * min_samples_req / (maj_samples + min_samples_req)
        # therefore, min_samples_req = self.p*maj_samples/(100 - self.p)
        total_min_samples_reqd = math.ceil(self.p * num_maj_samples / (100 - self.p))
        extra_min_samples_reqd = total_min_samples_reqd - num_min_samples

        # pick random minority samples to resample using SMOTE
        resample_indices = np.random.randint(
            0, high=num_min_samples, size=extra_min_samples_reqd
        )

        # iterate over chosen minority samples
        smoted_samples = []
        for resample_index in resample_indices:
            # get SMOTE sample by passing the minority sample
            # and the index of sample in minority list
            sample_neighbours = self.nn[resample_index]
            random_sample = self.X_min[resample_index]
            smoted_samples.append(
                self.get_synthetic_sample(random_sample, sample_neighbours)
            )
        # create a numpy array from resampled minority examples
        # and corresponding labels
        smoted_samples = np.array(smoted_samples)
        smoted_labels = np.array(
            [self.minority_label for _ in range(extra_min_samples_reqd)]
        )

        # create full sample and labels combining majority, minority and smoted samples
        X_resampled = np.concatenate((self.X_maj, self.X_min, smoted_samples), axis=0)
        y_resampled = np.concatenate((self.y, smoted_labels), axis=0)

        # shuffle
        if shuffle is True:
            np.random.shuffle(X_resampled)
            np.random.shuffle(y_resampled)

        return X_resampled, y_resampled
