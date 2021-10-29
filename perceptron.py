import numpy as np
from numpy import ndarray


def sigmoid(x):
    """
        Sigmoid activation function
    Args:
        x (ndarray, int): Value to be processed
    Returns:
        y (ndarray, int): Output
    """
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
        Sigmoid activation function
    Args:
        x (ndarray, int): Value to be processed
    Returns:
        y (ndarray, int): Output
    """
    return x * (1.0 - x)


def mse(x: ndarray, y: ndarray):
    """
        Mean Squared Error loss function
    Args:
        x (ndarray): The ground trut
        y (ndarray): The predicted values
    Returns:
        (float): Output
    """
    return np.average((x - y) ** 2)


class MLP:
    """
    A Multilayer Perceptron

    Args:
        num_inputs (int): numer of imputs layer
        hidden_layers (list) : list of neutron layers. Number in list specifying a number of neutron in hidden layer
        num_outputs (int): number of output layer

    Attributes:
        weights (list): list of weights for all connection between layers
        activations (list):
    Returns:
        None
    Example:

    """

    def __init__(self, num_inputs: int, hidden_layers: list, num_outputs: int):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]

        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        self.derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            self.derivatives.append(d)

        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)

    def forward_propagate(self, inputs, verbose=False) -> ndarray:
        """
            Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
            verbose (bool): Enable verbose output.
        Returns:
            activations (ndarray): Output values
        """
        activations = inputs
        self.activations[0] = activations
        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = sigmoid(net_inputs)
            self.activations[i + 1] = activations
        if verbose:
            print("activations units: \n {}".format(activations))
        return activations

    def back_propagate(self, error: ndarray, verbose: bool = False):
        """
            Backpropogates an error signal.

        Args:
            error (ndarray): The error to backpropagation.
            verbose (bool): Enable verbose output.

        """
        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activations = self.activations[i + 1]
            delta = error * sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations, delta_reshaped)
            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)
            if verbose:
                print("Derivatives for W{}: \n{}".format(i, self.derivatives))

    def train(self, x: ndarray, y: ndarray, epochs: int, learning_rate: float, verbose = False):
        """Trains model running forward prop and backprop
        Args:
            x (ndarray): Training data.
            y (ndarray): Training data.
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0
            for j, inputs in enumerate(x):
                target = y[j]
                # activate the network
                outputs = self.forward_propagate(inputs, verbose=verbose)
                error = target - outputs

                self.back_propagate(error, verbose=verbose)
                # now perform gradient descent on the derivatives
                # (will update the weights)
                self.gradient_descent(learning_rate, verbose=verbose)
                # keep track of the MSE
                sum_errors += mse(target, outputs)
            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(y), i + 1))
        print("Training complete!")
        print("=====")

    def gradient_descent(self, learning_rate: float, verbose=False) -> None:
        """
            Learns by descending the gradient

        Args:
            learning_rate (float): How fast to learn.
            verbose (bool): Enable verbose output.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weight = self.weights[i]
            if verbose:
                print("Original W{}: \n{}".format(i, weight))
            derivative = self.derivatives[i]
            weight += derivative * learning_rate
            if verbose:
                print("upadated W{}: \n{}".format(i, weight))

    def predict(self, x):
        """
        Predict for samples in X.
        Args:
            x (ndarray): Training data.
        """

        outputs = self.forward_propagate(x)
        print("Our network believes that is equal to {}".format(outputs))
        return outputs


