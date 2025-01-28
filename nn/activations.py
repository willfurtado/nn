"""
Defines common activation functions used in models
"""

import numpy as np


def relu_activation(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rectified Linear Unit (ReLU) Activation function

    Parameters:
        s (np.ndarray): Layer output of shape (B, layer_dim)

    Returns:
        (tuple[np.ndarray, np.ndarray]): Two-element tuple of ReLU activation output,
            of shape (B, layer_dim) and ReLU gradient information, of shape (B, layer_dim)
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
        x (np.ndarray): Layer output of shape (B, layer_dim)
        eps (float): Small epsilon used for numerical stability. Defaults to 10e-15.

    Returns:
        (tuple[np.ndarray, np.ndarray]): Two-element tupel of sigmoid activation output,
            of shape (B, layer_dim) and sigmoid gradient information, of shape (B, layer_dim)
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
        x (np.ndarray): Layer output of shape (B, layer_dim)

    Returns:
        (np.ndarray): Sigmoiod output of shape (B, layer_dim)
    """
    out_arr_pos, out_arr_neg = np.zeros(x.shape), np.zeros(x.shape)

    np.exp(-x, where=x >= 0, out=out_arr_pos)
    out_arr_pos = 1.0 / (1.0 + out_arr_pos)
    out_arr_pos = np.where(out_arr_pos == 1.0, 0, out_arr_pos)

    np.exp(x, where=x < 0, out=out_arr_neg)

    out_arr_neg = out_arr_neg / (1.0 + out_arr_neg)

    return out_arr_pos + out_arr_neg
