import numpy as np


class MLP:
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs: int = 4, hidden_layers: list = None, num_outputs: int = 4) -> None:
        if hidden_layers is None:
            hidden_layers = [2]
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        layers = [self.num_outputs] + self.hidden_layers + [self.num_outputs]
        self.weights = []
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(weight)
            print(weight)
        print(self.weights)


w = MLP(4, [2], 4)
