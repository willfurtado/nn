"""
Fully-connected network model implementation
"""

from typing import Callable

import numpy as np
from nn.models.base import Model
from nn import activations


class FullyConnectedNetwork(Model):
    """
    Fully connected neural network implemented using NumPy
    """

    valid_activations: list[str, callable] = {
        "relu": activations.relu_activation,
        "sigmoid": activations.sigmoid_activation,
    }

    def __init__(
        self,
        layer_dimensions: list[int],
        activations: list[Callable],
        loss_fn: Callable,
        init_method: str = "gaussian",
    ) -> None:
        """
        Construct a `FullyConnectedNetwork` instance with given architecture

        Parameters:
            layer_dimensions (list[int]): List of layer sizes to use in the network
            activations (list[Callable): List of activation functions to use in the network
            loss_fn (Callable): Loss function to use in the network
            init_method (str): Model weight initialization method. Defaults to `gaussian`.

        Returns:
            (FullyConnectedNetwork): Returns an instance of the `FullyConnectedNetwork` class
        """
        self.layer_dimensions = layer_dimensions
        self.activations = activations
        self.loss = loss_fn

        self._initialize_weight_matrices(how=init_method)
        self._initialize_bias_vectors(how=init_method)

    def forward_pass(
        self, X_batch: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """
        Computes an entire forward pass on the network

        Parameters:
            X_batch (np.ndarray): Input batch of shape (B, input_dim)

        Returns:
            (tuple[np.ndarray, list[dict[str, np.ndarray]]]): Two-element tuple containing raw model output and
                intermediate model activation cache used for backward pass
        """
        output = X_batch
        layer_caches = []
        for W, b, activation_fn in zip(self.W, self.b, self.activations):

            activation = self.valid_activations.get(activation_fn)

            if activation is None:
                raise ValueError(f"Activation function: {activation_fn} not supported")

            output, layer_cache = self._forward(output, W, b, activation)
            layer_caches.append(layer_cache)

        return output, layer_caches

    def backward_pass(
        self,
        dL_dg: np.ndarray,
        layer_caches: list[dict[str, np.ndarray]],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Computes one backward pass of the network

        Parameters:
            dL_dg (np.ndarray): Gradient of loss function w.r.t. last layer output
            layer_caches (list[dict[str, np.ndarray]]): Intermediate model activations, cached from forward pass

        Returns:
            (tuple[list[np.ndarray], list[np.ndarray]]): Two-element tuple containing list of weight gradients
                for each layer and list of bias gradients for each layer
        """
        final_layer_cache = layer_caches[-1]
        final_layer_delta = np.multiply(dL_dg, final_layer_cache["grad_act_fn"])

        final_layer_W = final_layer_cache["W"]
        final_layer_dW = np.mean(
            np.einsum("ij,ik->ikj", final_layer_delta, final_layer_cache["x"]),
            axis=0,
        )
        final_layer_db = np.mean(final_layer_delta[:, None, :], axis=0)

        grad_Ws = [final_layer_dW]
        grad_bs = [final_layer_db]

        prev_delta = final_layer_delta
        prev_w = final_layer_W

        num_layers = len(layer_caches)
        for l in reversed(range(num_layers - 1)):
            cache_l = layer_caches[l]

            curr_delta = np.multiply(
                cache_l["grad_act_fn"],
                np.einsum("ij,jk->ki", prev_w, prev_delta.T),
            )
            curr_dW = np.mean(np.einsum("ij,ik->ikj", curr_delta, cache_l["x"]), axis=0)
            curr_db = np.mean(curr_delta[:, None, :], axis=0)

            grad_Ws.append(curr_dW)
            grad_bs.append(curr_db)

            prev_delta = curr_delta
            prev_w = cache_l["W"]

        return grad_Ws[::-1], grad_bs[::-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Pass new array of data points through the network, discarding gradient information

        Parameters:
            X (np.ndarray): Input batch of data of shape (B, input_shape)

        Returns:
            (np.ndarray): Raw model output of shape (B, num_classes)
        """
        output, _ = self.forward_pass(X)

        return output

    def _forward(
        self,
        x: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        activation_fn: Callable,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Computes one layer's forward pass and caches gradient information

        Parameters:
            x (np.ndarray): Input data point of shape (B, input_dim)
            W (np.ndarray): Model weight matrix of shape (input_dim, output_dim)
            b (np.ndarray): Model bias vector shape (1, output_dim)
            activation_fn (Callable): Activation function to call on linear output

        Returns:
            (tuple[np.ndarray, dict[str, np.ndarray]]): Two-element tuple of layer output and
                dictionary caching activation and gradient information
        """
        out, grad_act_fn = activation_fn(x @ W + b)
        cache = dict(x=x, W=W, b=b, grad_act_fn=grad_act_fn)

        return out, cache

    def _initialize_weight_matrices(self, how: str = "gaussian") -> None:
        """
        Creates a list of weight matrices representing model weights

        Parameters:
            how (str): Weight initialization method. Defaults to "gaussian".

        Returns:
            (None): The `W` attribute of the model will be updated inplace
        """
        if how != "gaussian":
            raise NotImplementedError(
                f"Weights initialization with {how} not yet supported"
            )

        num_layers = len(self.layer_dimensions) - 1
        self.W = [
            np.random.normal(
                loc=0,
                scale=0.01,
                size=(self.layer_dimensions[i], self.layer_dimensions[i + 1]),
            )
            for i in range(num_layers)
        ]

    def _initialize_bias_vectors(self, how: str = "gaussian") -> None:
        """
        Creates a list of bias vectors representing model biases

        Parameters:
            how (str): Bias initialization method. Defaults to "gaussian".

        Returns:
            (None): The `b` attribute of the model will be updated inplace
        """
        if how != "gaussian":
            raise NotImplementedError(
                f"Bias initialization with {how} not yet supported"
            )

        num_layers = len(self.layer_dimensions) - 1
        self.b = [
            np.random.normal(loc=0, scale=0.01, size=(1, self.layer_dimensions[i + 1]))
            for i in range(num_layers)
        ]

    def __repr__(self) -> str:
        """
        String representation of the `FullyConnectedNetwork` class
        """
        return f"{self.__class__.__name__}(layers={self.layer_dimensions})"
