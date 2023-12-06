# Nischal, Sakshi
# 1002_085_832
# 2023_03_18
# Assignment_02_01
import numpy as np
import tensorflow as tf

def loss_function(true_val, pred_val, loss):
  # Calculate SVM Loss
  if loss.lower() == "svm":
    svm_loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - true_val * pred_val))
    return svm_loss
  # Calculate Mean Squared Error 
  if loss.lower() == "mse":
    mse_loss = tf.reduce_mean(tf.square(pred_val - true_val))
    return mse_loss
  # Calculate Cross Entropy Loss
  if loss.lower() == "cross_entropy":
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_val, labels=true_val))
    return cross_entropy_loss

def activation_function(X, activations):
  # Linear Activation Function
  if activations.lower() == "linear":
    linear_activation = X
    return linear_activation
  # Sigmoid Activation Function
  if activations.lower() == "sigmoid":
    sigmoid_activation = tf.nn.sigmoid(X)
    return sigmoid_activation
  # Re-LU Activation Function
  if activations.lower() == "relu":
    relu_activation = tf.nn.relu(X)
    return relu_activation

def split_data(X_train,Y_train,validation_split):  
  # Splitting training data into train and validation set
  start_point = int(validation_split[0] * X_train.shape[0])
  end_point = int(validation_split[1] * X_train.shape[0])
  train_x = np.concatenate((X_train[:start_point], X_train[end_point:]))
  train_y = np.concatenate((Y_train[:start_point], Y_train[end_point:]))
  val_x = X_train[start_point:end_point]
  val_y = Y_train[start_point:end_point]

  return train_x, train_y, val_x, val_y

def forward_propagation(X_train, Y_train, weights, activations, loss, layers):
  # The function will return the error and final output of the neural network
  activated_output_list = [X_train]
  for i in range(len(layers)):
    # Calculating XW for all layers
    z = tf.matmul(activated_output_list[-1], weights[i])
    predicted_value = activation_function(z,activations[i])
    # Add array of ones to the activated output for all layers except last layer
    if i != (len(layers)-1):
      ones_array = tf.ones((tf.shape(predicted_value)[0], 1), dtype=predicted_value.dtype)
      predicted_value = tf.concat([ones_array, predicted_value], axis=1)
    activated_output_list.append(predicted_value)
  # Calculate error for actual y-value and predicted y-value
  error = loss_function(Y_train, activated_output_list[-1], loss)
  return [error,  activated_output_list[-1]]

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",validation_split=[0.8,1.0],weights=None,seed=2):
  error_per_epoch_list = []
  # Initialize weights for each layer
  if weights == None:
    # Random Initialization - If weights are passed as None
    weight = []
    for i in range (len(layers)):
      if i==0:
        np.random.seed(seed)
        w = tf.Variable(np.random.randn(X_train.shape[1]+1,layers[i]).astype(np.float32))
        weight.append(w)
      else:
        np.random.seed(seed)
        w = tf.Variable(np.random.randn(layers[i-1]+1, layers[i]).astype(np.float32))
        weight.append(w)
  else:
    # Take weights from function parameters
    weight = []
    for w in weights:
      w = tf.Variable(w.astype(np.float32))
      weight.append(w)

  # Splitting training and testing data using validation split
  X_train, Y_train, X_test, Y_test = split_data(X_train,Y_train,validation_split)
  # Adding bias to training and testing data
  X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
  X_train = tf.cast(X_train, tf.float32)
  X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
  X_test = tf.cast(X_test, tf.float32)

  # Training network
  for epoch in range(epochs):
    for i in range (0, len(X_train), batch_size):
      # Extracting data into batches
      X_batch = X_train[i : i+batch_size]
      X_batch = tf.cast(X_batch, tf.float32) 
      Y_batch = Y_train[i : i+batch_size]
      Y_batch = tf.cast(Y_batch, tf.float32) 
      # Calculate Gradient
      with tf.GradientTape() as tape:
        tape.watch(weight)
        error_per_batch = forward_propagation(X_batch, Y_batch, weight, activations, loss, layers)[0]
      de_dw = tape.gradient(error_per_batch, weight)
      # Update Weights
      for i  in range(len(weight)):
        weight[i] = weight[i] - (alpha * de_dw[i])
    # If there is any data left
    if X_train.shape[0] % batch_size != 0:
      X_batch =  X_train[-(X_train.shape[0] % batch_size):]
      Y_batch = Y_train[-(X_train.shape[0] % batch_size):]
      # Calculate Gradient
      with tf.GradientTape() as tape:
        tape.watch(weight)
        error_per_batch  = forward_propagation(X_batch, Y_batch, weight, activations, loss, layers)[0]
      de_dw = tape.gradient(error_per_batch,weight)
      # Update Weights
      for i  in range(len(weight)):
        weight[i] = weight[i] - (alpha * de_dw[i])
    # Calculating error for validation set while network is frozen
    test_error = forward_propagation(X_test, Y_test, weight, activations, loss, layers)[0]
    error_per_epoch_list.append(test_error)
  
  # Calculating actual output for multilayer neural network when validation set is used as input
  actual_output = forward_propagation(X_test, Y_test, weight, activations, loss, layers)[1]

  return [weight, error_per_epoch_list, actual_output]