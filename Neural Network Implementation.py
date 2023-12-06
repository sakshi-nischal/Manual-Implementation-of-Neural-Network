# Nischal, Sakshi
# 1002_085_832
# 2023_02_27
# Assignment_01_01
import numpy as np

def calculate_sigmoid(x):
  return 1 / (1 + np.exp(-x))

def calculate_mse(Y_pred, Y_true):
  return np.mean((Y_pred - Y_true)**2)

def calculate_mse_list_and_output(X, Y, weights, layers):
  # Dot product of weights and training data
  activations = [X]
  count=0
  for w in range(len(layers)):
    count+=1
    layer_input = activations[-1]
    layer_output = np.dot(weights[w], layer_input)
    activation = calculate_sigmoid(layer_output)
    if(count!=len(layers)):
      activation = np.vstack((np.ones((1, activation.shape[1])), activation))
    activations.append(activation)
  return calculate_mse(activations[-1], Y)

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
  weights = []
  mse_list = []
  # Initializing weights including bias
  for i in range(len(layers)):
    # Weights for first layer including bias
    if i==0:
      np.random.seed(seed)
      w = np.random.randn(layers[i],X_train.shape[0]+1)
      weights.append(w)
    # Weights for other layers including bias
    else:
      np.random.seed(seed)
      w = np.random.randn(layers[i], layers[i-1]+1)
      weights.append(w)
  # Adding bias to training and testing data
  X_train = np.vstack((np.ones((1, X_train.shape[1])), X_train))
  X_test = np.vstack((np.ones((1, X_test.shape[1])), X_test))
  temp_weight_array = []
  for w in weights:
      temp_weights = np.copy(w)
      temp_weight_array.append(temp_weights)
  
  # Training the Network
  for i in range(epochs):
    for i in range(X_train.shape[1]):
      X_train_sample = X_train[:,i:i+1]
      Y_train_sample = Y_train[:,i:i+1]
      for i, w1 in enumerate(weights):
        for j, w2 in enumerate(w1):
          for k, w in enumerate(w2):
            weights[i][j][k] = weights[i][j][k]
            # Adding StepSize and Calculating mean square error
            weights[i][j][k] = weights[i][j][k] + h
            add_step_size = calculate_mse_list_and_output(X_train_sample, Y_train_sample, weights, layers)
            weights[i][j][k] = weights[i][j][k] - h
            # Subtracting StepSize and Calculating mean square error
            weights[i][j][k] = weights[i][j][k] - h
            subtract_step_size = calculate_mse_list_and_output(X_train_sample, Y_train_sample, weights, layers)
            weights[i][j][k] = weights[i][j][k] + h
            # Calculate gradient
            derivative = (add_step_size - subtract_step_size)/(2*h)
            # Update weight using calculated derivative
            new_weight = weights[i][j][k] - (alpha*derivative)
            temp_weight_array[i][j][k] = new_weight

      weights = []
      for i in temp_weight_array:
        layer_w = np.copy(i)
        weights.append(layer_w)
    
    # Calculating mean square error for testing data
    mse_per_epoch = []
    for i in range(X_test.shape[1]):
      X_test_sample = X_test[:,i:i+1]
      Y_test_sample = Y_test[:,i:i+1]
      mse_per_epoch.append(calculate_mse_list_and_output(X_test_sample, Y_test_sample, weights, layers))
    mse_list.append(np.mean(mse_per_epoch))
  
  # Calculate final output of Neural Network
  output_last_layer = []
  for w in range(len(layers)):
    if len(output_last_layer) == 0:
      Y = np.dot(temp_weight_array[w], X_test)
      output_last_layer.append(calculate_sigmoid(Y))
    else:
      output = np.vstack((np.ones((1, output_last_layer[-1].shape[1])), output_last_layer[-1]))
      Y = np.dot(temp_weight_array[w], output)
      output_last_layer.append(calculate_sigmoid(Y))
  
  weights = temp_weight_array
  return [weights, mse_list, output_last_layer[-1]]