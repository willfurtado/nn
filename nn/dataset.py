"""
Organizes datasets using object-oriented approach
"""

import numpy as np


class Dataset:
    """
    Dataset class to organize data
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Creates an instance of the `Dataset` class

        Parameters:
            x (np.ndarray):
            y (np.ndarray):
        """
        self.x = x
        self.y = y

    def create_batches(
        self,
        batch_size: int,
        drop_last: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Splits training data into batches
        """
        shuffled_idx = np.random.permutation(len(self.x))
        self.x = self.x[shuffled_idx]
        self.y = self.y[shuffled_idx]

        remainder = int(self.x.shape[0] % batch_size)
        first_X_train, last_X_train = (
            self.x[:-remainder, :],
            self.y[-remainder:, :],
        )
        first_y_train, last_y_train = self.y[:-remainder], self.y[-remainder:]

        num_batches = np.ceil(first_X_train.shape[0] / batch_size)

        X_batches = np.array_split(first_X_train, num_batches)
        y_batches = np.array_split(first_y_train, num_batches)

        if not drop_last:
            X_batches = X_batches + [last_X_train]
            y_batches = y_batches + [last_y_train]

        return X_batches, y_batches
