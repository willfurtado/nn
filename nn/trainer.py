"""
Module responsible for model training
"""

import numpy as np
from nn.models.base import Model
from nn.dataset import Dataset
from nn.loss import logistic_loss


class Trainer:
    """
    Houses all logic to train models
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        drop_last: bool = False,
        verbose: bool = False,
    ):
        """
        Creates an instance of the `Trainer` class
        """
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.drop_last = drop_last
        self.verbose = verbose

        self.losses, self.accuracies = [], []

    def train(self) -> None:
        """
        Launches a training run for `num_epochs`
        """
        for epoch in range(1, self.num_epochs + 1):
            print("-" * 15)
            print(f"Epoch: {epoch}")
            print("-" * 15)
            self._run_epoch()

    def _run_epoch(self) -> None:
        """
        Runs one epoch of training
        """
        X_batches, y_batches = self.dataset.create_batches(
            batch_size=self.batch_size, drop_last=self.drop_last
        )

        losses_epoch, accuracies_epoch = [], []

        for X_batch, y_batch in zip(X_batches, y_batches):
            output, layer_caches = self.model.forward_pass(X_batch)

            accuracy = np.mean(np.squeeze(output > 0.5) == y_batch)
            accuracies_epoch.append(accuracy)

            loss, dL_dg = logistic_loss(output, y_batch)
            losses_epoch.append(np.mean(loss))

            grad_Ws, grad_bs = self.model.backward_pass(dL_dg, layer_caches)
            self.model.W = [
                W - self.learning_rate * grad_W
                for grad_W, W in zip(grad_Ws, self.model.W)
            ]
            self.model.b = [
                b - self.learning_rate * grad_b
                for grad_b, b in zip(grad_bs, self.model.b)
            ]

        print(f"Average Training Loss: {np.mean(losses_epoch)}")
        print(f"Average Training Accuracy: {np.mean(accuracies_epoch)}\n")

        self.losses.extend(losses_epoch)
        self.accuracies.extend(accuracies_epoch)
