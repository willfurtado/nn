"""
Definitions for common loss functions used during model trainign
"""

import numpy as np


def logistic_loss(
    g: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the loss and gradient for binary classification with logistic loss

    Parameters:
        g (np.ndarray): Output of final layer with sigmoid activation, of shape (n, 1)
        y (np.ndarray): Vector of labels, of shape (n,) where y[i] is the label for x[i] and y[i] in {0, 1}

    Returns:
        (tuple[np.ndarray, np.ndarray]): A two-element tuple, where the first is an array of loss values and
            the second is the gradient of the loss w.r.t. `g`
    """
    y = y.reshape(g.shape)
    loss = -np.log((g**y) * ((1 - g) ** (1 - y)))
    dL_dg = ((1 - y) / (1 - g)) - (y / g)

    assert (
        loss.shape == g.shape
    ), f"Loss shape mismatch in logistic loss. Got: {loss.shape} and {g.shape}"
    assert (
        loss.shape == y.shape
    ), f"Loss shape mismatch in logistic. Got: {loss.shape} and {y.shape}"

    return loss, dL_dg
