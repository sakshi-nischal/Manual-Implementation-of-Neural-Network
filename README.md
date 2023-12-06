# Manual-Implementation-of-Neural-Network
This Python script implements a simple neural network with backpropagation for training. The key functionalities include calculating the sigmoid function, mean squared error (MSE), and training a multi-layer neural network.

Functions:
1. calculate_sigmoid(x)
Calculates the sigmoid function for a given input x.

2. calculate_mse(Y_pred, Y_true)
Calculates the Mean Squared Error (MSE) between predicted Y_pred and true Y_true values.

3. calculate_mse_list_and_output(X, Y, weights, layers)
Calculates the MSE for a given set of inputs X, true values Y, weights, and network layers.

4. multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2)
Trains a multi-layer neural network with backpropagation.
X_train, Y_train: Training data and corresponding true values.
X_test, Y_test: Testing data and corresponding true values.
layers: List specifying the number of neurons in each layer.
alpha: Learning rate.
epochs: Number of training epochs.
h: Small value for numerical gradient approximation (default is 0.00001).
seed: Random seed for weight initialization (default is 2).
