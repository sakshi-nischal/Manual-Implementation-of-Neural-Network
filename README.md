# Manual-Implementation-of-Neural-Network
## Neural Network Implementation
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

## Neural Network Implementation using TensorFlow
This Python script implements a multi-layer neural network using TensorFlow. The network includes functionalities for forward propagation, activation functions, loss functions, and training using backpropagation.

Functions:
1. loss_function(true_val, pred_val, loss)
Calculates the loss based on the specified loss function.
Supports "svm" (Support Vector Machine), "mse" (Mean Squared Error), and "cross_entropy" (Cross-Entropy) losses.

2. activation_function(X, activations)
Applies the specified activation function to the input.
Supports "linear," "sigmoid," and "relu" activation functions.

3. split_data(X_train, Y_train, validation_split)
Splits the training data into training and validation sets based on the specified validation split ratio.

4. forward_propagation(X_train, Y_train, weights, activations, loss, layers)
Performs forward propagation to calculate the error and final output of the neural network.

5. multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="svm", validation_split=[0.8, 1.0], weights=None, seed=2)
Trains a multi-layer neural network using TensorFlow.
Supports customizable parameters such as learning rate (alpha), batch size (batch_size), number of epochs (epochs), loss function (loss), and weight initialization (weights).
Returns the trained weights, error per epoch, and actual output of the network on the validation set.
