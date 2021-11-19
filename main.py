import numpy as np

from perceptron import MLP

if __name__ == "__main__":
    # create a dataset to train a network for the sum operation

    # create a dataset to train a network for the sum operation
    x_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(4, [2,4], 4, bias = True)

    # train network
    x = mlp.train(x_train, y_train, 5000, 0.2, verbose=False)
    mlp.plot_mse('bias')
    x_test = np.array([1, 0, 0, 0])

    # get a prediction
    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 1, 0, 0])

    # get a prediction
    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 0, 1, 0])

    # get a prediction
    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 0, 0, 1])

    # get a prediction
    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
