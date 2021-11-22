import numpy as np
import pandas as pd
from perceptron import MLP
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # create a dataset to train and test a network 
    x_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(4, [2], 4, bias=True)

    # train network
    x = mlp.train(x_train, y_train, 1, 0.2, verbose=True)
    # plot a MSE for each outputs and avg output
    mlp.plot_mse('example_plot')

    # plot activations on hidden layer each training session
    # list_epochs = [i+1 for i in range(1000)]
    # df_hidden = pd.DataFrame(mlp.hidden_activates, columns=['1 neuron', '2 neuron'])
    # df_epoch = pd.DataFrame(list_epochs, columns=['epochs'])
    # df = pd.concat([df_epoch, df_hidden], axis=1)
    # plt.clf()
    # sns.lineplot(x='epochs', y='value', hue='variable', 
    #          data=pd.melt(df, ['epochs']),legend=False)
    # plt.legend(title = "Hidden_layer")
    # plt.savefig('hidden.png')
    
    # get first prediction
    x_test = np.array([1, 0, 0, 0])
    outputs = mlp.predict(x_test)
    print("Network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 1, 0, 0])

    # get second prediction
    outputs = mlp.predict(x_test)
    print("Network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 0, 1, 0])
    # get third prediction

    outputs = mlp.predict(x_test)
    print("Network believes that is equal to {}".format(outputs))
    x_test = np.array([0, 0, 0, 1])
    # get fourth prediction

    outputs = mlp.predict(x_test)
    print("Network believes that is equal to {}".format(outputs))
