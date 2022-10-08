from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import ArrayLike


class NeuralNetwork:
    def __init__(
        self,
        layer_dimensions: List[int],
        activations: List[Callable],
        loss_fn: Callable,
        init_method: str = "gaussian",
    ) -> None:
        """
        Construct a NeuralNetwork instance with given architecture

        Parameters:
            layer_dimensions (List[int]):
            activations (List[Callable):
            loss_fn (Callable):

        Returns:
            (NeuralNetwork): Returns an instance of the NeuralNetwork class
        """
        self.layer_dimensions = layer_dimensions
        self.activations = activations
        self.loss = loss_fn

        self._initialize_weight_matrices(how=init_method)
        self._initialize_bias_vectors(how=init_method)

    def _initialize_weight_matrices(self, how: str = "gaussian") -> None:
        """
        Creates a list of weight matrices defining the weights of NN

        Parameters:
            layer_dims: A list whose size is the number of layers. layer_dims[i]
                    defines the number of neurons in the i+1 layer.

        Returns:
            (None): A list of weight matrices
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

        if how != "gaussian":
            raise NotImplementedError(
                f"Bias initialization with {how} not yet supported"
            )

        num_layers = len(self.layer_dimensions) - 1
        self.b = [
            np.random.normal(
                loc=0, scale=0.01, size=(1, self.layer_dimensions[i + 1])
            )
            for i in range(num_layers)
        ]

    def _forward(
        self,
        x: ArrayLike,
        W: ArrayLike,
        b: ArrayLike,
        activation_fn: Callable,
    ) -> Tuple[ArrayLike, Dict]:
        """Computes one layer forward calculation for inputs"""
        out, grad_act_fn = activation_fn(x @ W + b)
        cache = dict(x=x, W=W, b=b, grad_act_fn=grad_act_fn)
        return out, cache

    def _forward_pass(
        self, X_batch: ArrayLike
    ) -> Tuple[ArrayLike, List[ArrayLike]]:
        """Computes an entire forward pass on the network"""
        output = X_batch
        layer_caches = []
        for W, b, activation_fn in zip(self.W, self.b, self.activations):

            activation = self.valid_activations.get(activation_fn)

            if activation is None:
                raise ValueError(
                    f"Activation function: {activation_fn} not supported"
                )

            output, layer_cache = self._forward(output, W, b, activation)
            layer_caches.append(layer_cache)

        return output, layer_caches

    def _backward_pass(
        dL_dg: ArrayLike,
        layer_caches: List[ArrayLike],
    ) -> ArrayLike:
        """Computes one backward pass of the network"""
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
            curr_dW = np.mean(
                np.einsum("ij,ik->ikj", curr_delta, cache_l["x"]), axis=0
            )
            curr_db = np.mean(curr_delta[:, None, :], axis=0)

            grad_Ws.append(curr_dW)
            grad_bs.append(curr_db)

            prev_delta = curr_delta
            prev_w = cache_l["W"]

        return grad_Ws[::-1], grad_bs[::-1]

    def train(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        num_epochs: int,
        learning_rate: float,
    ) -> Tuple[List[float], List[float]]:
        """Train the Neural Network"""
        losses, accuracies = [], []
        for epoch in range(1, self.num_epochs + 1):
            print("-" * 15)
            print(f"Epoch: {epoch}")
            print("-" * 15)
            X_batches, y_batches = self.create_batches(X, y)

            losses_epoch, accuracies_epoch = [], []

            for X_batch, y_batch in zip(X_batches, y_batches):
                output, layer_caches = self._forward_pass(
                    X_batch, self.W, self.b, self.activations
                )

                accuracy = np.mean(np.squeeze(output > 0.5) == y_batch)
                accuracies_epoch.append(accuracy)

                loss, dL_dg = self.logistic_loss(output, y_batch)
                losses_epoch.append(np.mean(loss))

                grad_Ws, grad_bs = self._backward_pass(dL_dg, layer_caches)
                self.W = [
                    W - learning_rate * grad_W
                    for grad_W, W in zip(grad_Ws, self.W)
                ]
                self.b = [
                    b - learning_rate * grad_b
                    for grad_b, b in zip(grad_bs, self.b)
                ]

            print(f"Average Training Loss: {np.mean(losses_epoch)}")
            print(f"Average Training Accuracy: {np.mean(accuracies_epoch)}\n")

            losses.extend(losses_epoch)
            accuracies.extend(accuracies_epoch)

        return losses, accuracies

    def predict(self, X) -> ArrayLike:
        """Pass new array of data points through the network"""
        output, _ = self._forward_pass(X)

        return output

    @staticmethod
    def _stable_sigmoid(x: ArrayLike) -> ArrayLike:
        out_arr_pos, out_arr_neg = np.zeros(x.shape), np.zeros(x.shape)

        np.exp(-x, where=x >= 0, out=out_arr_pos)
        out_arr_pos = 1.0 / (1.0 + out_arr_pos)
        out_arr_pos = np.where(out_arr_pos == 1.0, 0, out_arr_pos)

        np.exp(x, where=x < 0, out=out_arr_neg)

        out_arr_neg = out_arr_neg / (1.0 + out_arr_neg)

        return out_arr_pos + out_arr_neg

    @staticmethod
    def sigmoid_activation(self, x: ArrayLike, eps: float = 10e-15):
        out = self._stable_sigmoid(x)
        grad = self._stable_sigmoid(x) * (1.0 - self._stable_sigmoid(x))
        return np.clip(out, a_min=eps, a_max=1 - eps), grad

    @staticmethod
    def logistic_loss(
        g: ArrayLike,
        y: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Computes the loss and gradient for binary classification with logistic
        loss

        Inputs:
        - g: Output of final layer with sigmoid activation,
             of shape (n, 1)

        - y: Vector of labels, of shape (n,) where y[i] is the label for x[i]
             and y[i] in {0, 1}

        Returns a tuple of:
        - loss: array of losses
        - dL_dg: Gradient of the loss with respect to g
        """
        y = y.reshape(g.shape)
        loss = -np.log((g**y) * ((1 - g) ** (1 - y)))
        dL_dg = ((1 - y) / (1 - g)) - (y / g)

        assert loss.shape == g.shape
        assert loss.shape == y.shape

        return loss, dL_dg

    @staticmethod
    def relu_activation(s: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        out = s * (s > 0)
        ds = 1.0 * (s > 0)

        assert out.shape == s.shape
        assert ds.shape == s.shape

        return out, ds

    @staticmethod
    def create_batches(
        X: ArrayLike,
        y: ArrayLike,
        batch_size: int = 100,
    ) -> Tuple[List[ArrayLike], List[ArrayLike]]:
        """Splits training data into batches"""
        shuffled_idx = np.random.permutation(len(X))
        X = X[shuffled_idx]
        y = y[shuffled_idx]

        remainder = int(X.shape[0] % batch_size)
        first_X_train, last_X_train = (
            X[:-remainder, :],
            X[-remainder:, :],
        )
        first_y_train, last_y_train = y[:-remainder], y[-remainder:]

        num_batches = np.ceil(first_X_train.shape[0] / batch_size)

        X_batches = np.array_split(first_X_train, num_batches) + [last_X_train]
        y_batches = np.array_split(first_y_train, num_batches) + [last_y_train]

        return X_batches, y_batches
