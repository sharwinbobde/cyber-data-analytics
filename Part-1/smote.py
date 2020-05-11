import pandas as pd
import numpy as np
import math


class SMOTE:
    def __init__(self, p: float, k: int, random_state: int = 1337) -> None:
        self.p = p
        self.k = k
        self.nn = None
        self.X_min = None
        self.X_maj = None
        self.y = None
        self.minority_label = None

        np.random.seed(random_state)

    def euclidean_distance(self, v1: np.array, v2: np.array) -> np.array:
        """
        Computes Euclidean distance between
        corresponding rows in `v1` and `v2`.
        """
        return np.sqrt(np.sum(np.square(v1 - v2), axis=1))

    def get_nearest_neighbours(
        self, sample: np.array, population: np.array
    ) -> np.array:
        """
        Computes `k` nearest-neighbours of `sample`
        present in array of vectors `population`

        Returns indices of `k` nearest-neighbours in `population`
        """
        # create copies of `sample` to compare to
        # every other sample in the population
        sample_duplicates = np.tile(sample, (population.shape[0], 1))
        distances = self.euclidean_distance(population, sample_duplicates)
        return np.argsort(distances)

    def get_minority_label(self, labels: np.array) -> (int, np.array):
        """
        Returns the label with lowest frequency in `labels` and
        boolean array indicating indices corresponding to minority labels
        """
        counts = np.bincount(labels)
        label = np.nonzero(counts)[0]
        label_counts = list(zip(label, counts[label]))
        label_counts.sort(key=lambda x: x[1])
        minority_label = label_counts[0][0]
        minority_label_map = labels == minority_label
        return minority_label, minority_label_map

    def get_synthetic_sample(self, sample: np.array, neighbours: np.array) -> np.array:
        """
        """
        nn_index = np.random.randint(0, high=self.k)
        nearest_neighbours = self.X_min[neighbours][1 : self.k + 1]
        nn_sample = nearest_neighbours[nn_index]
        weight = np.random.uniform(low=0, high=1)
        synthetic_sample = sample + (sample - nn_sample) * weight
        return synthetic_sample

    def fit(self, X, y):
        """
        """
        minority_label, minority_label_map = self.get_minority_label(y)
        X_min = X[minority_label_map, :]
        # if desired percentage for minority class < current ratio
        # raise an exception
        if self.p <= 100 * X_min.shape[0] / X.shape[0]:
            raise ValueError(
                f"""minority class in X already has a percentage of {round(100*X_min.shape[0]/X.shape[0], 2)} which is >= desired percentage self.p = {self.p}. This class is used to do oversampling of minority class, not undersampling"""
            )
        self.nn = np.apply_along_axis(self.get_nearest_neighbours, 1, X_min, X_min)
        self.minority_label = minority_label
        self.y = y
        self.X_min = X_min
        self.X_maj = X[~minority_label_map, :]

    def transform(self, shuffle=True):
        """
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

        if shuffle is True:
            np.random.shuffle(X_resampled)
            np.random.shuffle(y_resampled)

        return X_resampled, y_resampled


if __name__ == "__main__":
    df = pd.read_csv("./data/agg_feat.csv")
    columns = [
        "accountcode",
        "amount",
        "amount_eur",
        "bin",
        "card_id",
        "cardverificationcodesupplied",
        "countries_equal",
        "currencycode",
        "cvcresponsecode",
        "daily_avg_over_month",
        "day_of_week",
        "hour",
        "ip_id",
        "issuercountrycode",
        "mail_id",
        "prev_day_amount",
        "prev_day_perc_same_country",
        "prev_month_avg_amount",
        "prev_month_avg_amount_same_country",
        "prev_month_avg_amount_same_currency",
        "prev_month_perc_same_country",
        "prev_month_perc_same_currency",
        "prev_week_avg_amount",
        "shoppercountrycode",
        "shopperinteraction",
        "txid",
        "txvariantcode",
    ]
    X = df[columns].to_numpy()
    y = df["simple_journal"].to_numpy()
    smote = SMOTE(0.2, 6)
    smote.fit(X, y)
    X_resampled, y_resampled = smote.transform()
