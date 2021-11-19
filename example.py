import numpy as np

from perceptron import MLP

if __name__ == "__main__":
    
    # create a dataset to train and test a network 
    x_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(4, [2], 4)

    # train network
    x = mlp.train(x_train, y_train, 5000, 0.2)
    # plot a MSE for each outputs and avg output
    mlp.plot_mse('example_plot')
    
    
    # get first prediction
    x_test = np.array([1, 0, 0, 0])

    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 1, 0, 0])

    # get second prediction
    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 0, 1, 0])

    # get third prediction
    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 0, 0, 1])

    # get fourth prediction
    outputs = mlp.predict(x_test)
    print("Our network believes that is equal to {}".format(outputs))
