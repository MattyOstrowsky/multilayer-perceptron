import numpy as np
from numpy import ndarray

from tqdm import trange


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
        import numpy as np
        from perceptron import MLP
        from random import uniform

        x_train = np.array([[uniform(0, 10) for _ in range(2)] for _ in range(1000)])
        y_train = np.array([[(i[0] * i[1]**2)/2] for i in x_train])

        mlp = MLP(2, [2], 1)
        mlp.train(x_train, y_train, 50, 0.01, verbose=False)
    """

    def __init__(self, num_inputs: int, hidden_layers: list, num_outputs: int):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]
        self.bias = False
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

    def forward_propagate(self, inputs, bias, verbose=False) -> ndarray:
        """
            Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
            verbose (bool): Enable verbose output.
            bias (bool): Add bias to net_inputs.
        Returns:
            activations (ndarray): Output values
        """
        activations = inputs
        if verbose:
            print("Input values: \n {}".format(activations))
        self.activations[0] = activations
        for i, w in enumerate(self.weights):
            if bias:
                net_inputs = np.dot(activations, w)  
            else:
                net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations
        if verbose:
            print("\n****** Compute the weighted sums in each neuron, propagate results to the output layer ******")
            print("activations/output units: \n {}".format(activations))
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
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations, delta_reshaped)
            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)
            if verbose:
                print("Derivatives for W{}: \n{}".format(i, self.derivatives[i]))

    def train(self, x: ndarray, y: ndarray, epochs: int, learning_rate: float, bias: bool, verbose=False):
        """Trains model running forward prop and backprop
        Args:
            x (ndarray): Training data.
            y (ndarray): Training data.
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
            verbose (bool): Enable verbose output.
            bias (bool): Add bias to net_inputs.
        """
        # now enter the training loop
        self.bias = bias
        sum_errors_epochs = 0
        for i in trange(epochs):
            sum_errors = 0

            for j, inputs in enumerate(x):
                target = y[j]
                # activate the network
                outputs = self.forward_propagate(inputs, bias=bias, verbose=verbose)
                error = target - outputs
                if verbose:
                    print("\n ****** Error and derivatives calculation ******")
                    print("Error for each outputs: \n{}".format(error))
                self.back_propagate(error, verbose=verbose)
                # now perform gradient descent on the derivatives
                # keep track of the MSE
                sum_errors += self._mse(target, outputs)
                sum_errors_epochs += self._mse(target, outputs)
                # (will update the weights)
                self.gradient_descent(learning_rate, verbose=verbose)
                if verbose:
                    print("Training session {}/{}".format(j + 1, len(x)))
                    print('-' * 30)
            if verbose:
                print("\nAvg error: {} at epoch {}".format(sum_errors / len(y), i + 1))
                print("Epoch {} finished!!".format(i + 1))
                print("=" * 30)
                # Epoch complete, report the training error

        print("Training complete!")
        print("Avg error: {} after {} epochs".format(sum_errors_epochs / epochs, i + 1))
        print("=" * 30)

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
                print("\n ****** update layers with the value of the gradient! ******")
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

        Return:
            outputs (ndarray): Prediction for sample
        """

        outputs = self.forward_propagate(x, bias=self.bias)

        return outputs

    @staticmethod
    def _sigmoid(x):
        """
            Sigmoid activation function
        Args:
            x (ndarray, int): Value to be processed
        Returns:
            y (ndarray, int): Output
        """
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_derivative(x):
        """
            Sigmoid activation function
        Args:
            x (ndarray, int): Value to be processed
        Returns:
            y (ndarray, int): Output
        """
        return x * (1.0 - x)

    @staticmethod
    def _mse(x: ndarray, y: ndarray):
        """
            Mean Squared Error loss function
        Args:
            x (ndarray): The ground trut
            y (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((x - y) ** 2)
