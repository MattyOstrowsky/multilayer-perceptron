import numpy as np
from numpy import ndarray
from printer import print_under_other, print_in_line, print_header
from tqdm import trange
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class MLP:
    """
    A Multilayer Perceptron

    Args:
        num_inputs (int): numer of imputs layer
        hidden_layers (list) : list of neutron layers. Number in list specifying a number of neutron in hidden layer
        num_outputs (int): number of output layer
        bias (bool): Add bias to net_inputs 
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

    def __init__(self, num_inputs: int, hidden_layers: list, num_outputs: int, bias: bool= True):
        self.hidden_activates = []
        self.bias = bias
        self.list_mse_epochs = []
        self.avg_list_mse_epochs = []
        if self.bias:
            self.num_inputs = num_inputs
            self.hidden_layers = [i+1 for i in hidden_layers]
            self.num_outputs = num_outputs+1
        else:
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
            
        # update bias neuron to 1
        if self.bias:
            for i in range(len(layers)-1):
                self.activations[i+1][-1] = 1.
            
            
    def forward_propagate(self, inputs, verbose=False) -> ndarray:
        """
            Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
            verbose (bool): Enable verbose output.
            
        Returns:
            activations (ndarray): Output layer values
        """
        activations = inputs
        if verbose:
            print_under_other("Input values: ", activations)
        self.activations[0] = activations
        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            if self.bias:
                self.activations[i + 1][:-1] = activations[:-1]
            else:
                self.activations[i + 1] = activations

        if verbose:
            print("hidden layers: ")
            if self.bias:
                print(self.activations[1][:-1])
                x = self.activations[1][:-1]
                self.hidden_activates.append(x.tolist())
            else:
                print(self.activations[1])
                x = self.activations[1]
                self.hidden_activates.append(x.tolist())
            print_header(' Compute the weighted sums in each neuron, propagate results to the output layer ')
            print_under_other("Output units: ", activations)
            
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
            if self.bias:
                delta = error * self._sigmoid_derivative(self.activations[i + 1])
                current_activations = self.activations[i]
            else:
                delta = error * self._sigmoid_derivative(self.activations[i + 1])
                current_activations = self.activations[i]
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations, delta_reshaped)
            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)
            if verbose:
                print_in_line("Layer ", i+1)
                print_under_other("Derivatives: ", self.derivatives[i])

    def train(self, x: ndarray, y: ndarray, epochs: int, learning_rate: float, verbose=False):
        """Trains model running forward prop and backprop
        Args:
            x (ndarray): Training data.
            y (ndarray): Training data.
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
            verbose (bool): Enable verbose output.
        """
        # now enter the training loop
        sum_errors_epochs = 0
        self.epochs = epochs
        for i in trange(epochs, desc='Epochs'):
            sum_errors = 0
            list_mse_epochs = []
            for j, inputs in enumerate(x):
                target = y[j]
                if verbose:
                    print('-' * 100)
                    print("Training session {}/{}".format(j + 1, len(x)))
                    print('-' * 100)
                # activate the network
                outputs = self.forward_propagate(inputs, verbose=verbose)
                if self.bias:
                    target = np.append(target,1)
                error = target - outputs
                if verbose:
                    print_header(' Error and derivatives calculation ')
                    print_under_other("Error for each outputs: ", error)
                    
                self.back_propagate(error, verbose=verbose)
                # now perform gradient descent on the derivatives
                # keep track of the MSE
                sum_errors += self._mse(target, outputs)
                sum_errors_epochs += self._mse(target, outputs)
                if self.bias:
                    list_mse_epochs.append((target[:-1] - outputs[:-1]) ** 2 )
                else:
                    list_mse_epochs.append((target - outputs) ** 2 )
                # (will update the weights)
                self.gradient_descent(learning_rate, verbose=verbose)
                
            q = [float(sum(col))/len(col) for col in zip(*list_mse_epochs)]
            self.avg_list_mse_epochs.append(np.average(list_mse_epochs))
            #q.append(np.average(list_mse_epochs))
            self.list_mse_epochs.append(q)
           
            if verbose:
                print("-" * 100)
                print("Avg mse error: {} at epoch {}".format("%.2f" % (sum_errors / len(y)), i + 1))
                print("Epoch {} finished!!".format(i + 1))
                print("-" * 100)
                # Epoch complete, report the training error
        # print(self.list_mse_epochs)
        print("=" * 100)
        print("Training complete!")
        print("Avg mse error: {} after {} epochs".format("%.2f" % (sum_errors_epochs / epochs), i + 1))
        print("=" * 100)

    def gradient_descent(self, learning_rate: float, verbose=False) -> None:
        """
            Learns by descending the gradient

        Args:
            learning_rate (float): How fast to learn.
            verbose (bool): Enable verbose output.
        """
        # update the weights by stepping down the gradient
        if verbose:
            print_header(' update layers with the value of the gradient! ')
        for i in range(len(self.weights)):
            weight = self.weights[i]
            derivative = self.derivatives[i]
            weight += derivative * learning_rate
            
            if verbose:
                print_in_line("Layer ", i+1)
                print_under_other("Original weights: ", weight)
                print_under_other("upadated weights: ", weight)

    def predict(self, x):
        """
        Predict for samples in X.
        Args:
            x (ndarray): Training data.

        Return:
            outputs (ndarray): Prediction for sample
        """
        prediction = self.forward_propagate(x)
        if self.bias:
            return np.round(prediction[:-1], 2)
        else:
            return np.round(prediction, 2)
        
    def plot_mse(self, plot_name:str='mse_plot'):
        columns = [ str(i+1) for i in range(len(self.list_mse_epochs[0]))]
        
        list_epochs = [i+1 for i in range(self.epochs)]
        df_mse = pd.DataFrame(self.list_mse_epochs, columns=columns)
        df_epoch = pd.DataFrame(list_epochs, columns=['epochs'])
        df = pd.concat([df_epoch, df_mse], axis=1)
        columns.append('avg')
        sns.lineplot(x='epochs', y='value', hue='variable', 
             data=pd.melt(df, ['epochs']),legend=False)
       
        sns.lineplot(x=list_epochs, y=self.avg_list_mse_epochs, style=True, dashes=[(2,2)],legend=False)
        plt.legend(title = "Inputs", labels=columns)
    
        plt.savefig(plot_name+'.png')
        
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
