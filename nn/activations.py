"""
Defines common activation functions used in models
"""

import numpy as np


def relu_activation(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rectified Linear Unit (ReLU) Activation function

    Parameters:
        s (np.ndarray):

    Returns:
        (tuple[np.ndarray, np.ndarray]):
    """
    out = s * (s > 0)
    ds = 1.0 * (s > 0)

    assert (
        out.shape == s.shape
    ), f"Output shape mismatch in ReLU layer. Got: {out.shape} and {s.shape}"
    assert (
        ds.shape == s.shape
    ), f"Gradient shape mismatch in ReLU layer. Got: {ds.shape} and {s.shape}"

    return out, ds


def sigmoid_activation(
    x: np.ndarray, eps: float = 10e-15
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sigmoid Activation function

    Parameters:
        x (np.ndarray):
        eps (float):

    Returns:
        (tuple[np.ndarray, np.ndarray]):
    """
    out = _stable_sigmoid(x)
    grad = _stable_sigmoid(x) * (1.0 - _stable_sigmoid(x))

    return np.clip(out, a_min=eps, a_max=1 - eps), grad


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid activation function, used within `sigmoid_activation`

    Handles positive and negative input separately in order to minimize the numerical error
    caused by instability of floating point operations for very small / large values.

    Parameters:
        x (np.ndarray):

    Returns:
        (np.ndarray):
    """
    out_arr_pos, out_arr_neg = np.zeros(x.shape), np.zeros(x.shape)

    np.exp(-x, where=x >= 0, out=out_arr_pos)
    out_arr_pos = 1.0 / (1.0 + out_arr_pos)
    out_arr_pos = np.where(out_arr_pos == 1.0, 0, out_arr_pos)

    np.exp(x, where=x < 0, out=out_arr_neg)

    out_arr_neg = out_arr_neg / (1.0 + out_arr_neg)

    return out_arr_pos + out_arr_neg
