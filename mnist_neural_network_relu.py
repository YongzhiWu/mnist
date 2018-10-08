#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:29:12 2018

@author: wuyz
"""

import numpy as np
from mnist_data import fetch_trainset, fetch_testset

np.random.seed(0)

# Numbers of input_layer and output_layer
INPUT_NODE = 784
OUTPUT_NODE = 10

TRAINSET_SIZE = 60000

# Hyperparameters of MNIST Neural Network
LAYER1_NODE = 500
LAYER2_NODE = 300
LAYER3_NODE = 100
BATCH_SIZE = 100
learning_rate = 0.005
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000
WEIGHT_BASE = 0.028

train_acc = []
test_acc = []
loss_fun = []

def softmax(matrix):
    m, n = matrix.shape
    for i in range(m):
        matrix[i:i+1] = np.exp(matrix[i:i+1]) / np.sum(np.exp(matrix[i:i+1]))
    return matrix     

def train(X_train, y_train, train_images, train_labels, test_images, test_labels):
    X_b = np.ones([BATCH_SIZE, 1])
    weights1 = WEIGHT_BASE * np.random.randn(INPUT_NODE, LAYER1_NODE)
    biases1 = WEIGHT_BASE * np.ones([1, LAYER1_NODE])
    weights2 = WEIGHT_BASE * np.random.randn(LAYER1_NODE, LAYER2_NODE)
    biases2 = WEIGHT_BASE * np.ones([1, LAYER2_NODE])
    weights3 = WEIGHT_BASE * np.random.randn(LAYER2_NODE, LAYER3_NODE)
    biases3 = WEIGHT_BASE * np.ones([1, LAYER3_NODE])
    weights4 = WEIGHT_BASE * np.random.randn(LAYER3_NODE, OUTPUT_NODE)
    biases4 = WEIGHT_BASE * np.ones([1, OUTPUT_NODE])
    
    for global_step in range(TRAINING_STEPS):
        start = (global_step * BATCH_SIZE) % TRAINSET_SIZE
        end = start + BATCH_SIZE
        X_train_batch = X_train[start:end]
        y_train_batch = y_train[start:end]
        
        layer1_input = np.dot(X_train_batch, weights1) + np.dot(X_b, biases1)
        layer1 = np.maximum(0, layer1_input)
        layer2_input = np.dot(layer1, weights2) + np.dot(X_b, biases2)
        layer2 = np.maximum(0, layer2_input)
        layer3_input = np.dot(layer2, weights3) + np.dot(X_b, biases3)
        layer3 = np.maximum(0, layer3_input)
        z = np.dot(layer3, weights4) + np.dot(X_b, biases4)
        
        p = softmax(z)
        '''
        if global_step%100 == 0:
            loss = -np.mean(y_train_batch * np.log(p))
            loss_fun.append(loss)
            print("After {0} training step(s), cross_entropy_mean is {1}".format(global_step, loss))
        '''
        z_delta = y_train_batch - p
        layer3_delta = z_delta.dot(weights4.T) * (np.maximum(0, layer3_input) * 1.0 / layer3_input)
        layer2_delta = layer3_delta.dot(weights3.T) * (np.maximum(0, layer2_input) * 1.0 / layer2_input)
        layer1_delta = layer2_delta.dot(weights2.T) * (np.maximum(0, layer1_input) * 1.0 / layer1_input)
        
        weights4 += learning_rate * layer3.T.dot(z_delta)
        weights3 += learning_rate * layer2.T.dot(layer3_delta)
        weights2 += learning_rate * layer1.T.dot(layer2_delta)
        weights1 += learning_rate * X_train_batch.T.dot(layer1_delta)
        
        biases4 += learning_rate * X_b.T.dot(z_delta)
        biases3 += learning_rate * X_b.T.dot(layer3_delta)
        biases2 += learning_rate * X_b.T.dot(layer2_delta)
        biases1 += learning_rate * X_b.T.dot(layer1_delta)
        '''
        if global_step>0 and global_step < 2000 and (global_step%10 == 0):
            train_accuracy = evaluate(train_images, train_labels,  weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4)
            test_accuracy = evaluate(test_images, test_labels, weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4)
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
            print("After {0} training step(s), train_accuracy is {1}, test_accuracy is {2}".format(global_step, train_accuracy, test_accuracy))
        '''
    return weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4

def evaluate(X, y, weights1, biases1, weights2, biases2, weights3, biases3, weights4, biases4):
    X_b = np.ones([X.shape[0], 1])
    layer1 = np.maximum(0, (np.dot(X, weights1) + np.dot(X_b, biases1)) )
    layer2 = np.maximum(0, (np.dot(layer1, weights2) + np.dot(X_b, biases2)) )
    layer3 = np.maximum(0, (np.dot(layer2, weights3) + np.dot(X_b, biases3)) )
    z = np.dot(layer3, weights4) + np.dot(X_b, biases4)
    p = softmax(z)
    y_predict = []
    for row in p:
        y_predict.append((np.where(row == np.max(row)))[0][0])
    y_predict = np.array([y_predict]).T
    accuracy = ((np.array([y]).T) == y_predict).sum().astype(float) / len(y_predict)
    return accuracy

def pre_process(images, labels):
    labels_list = []
    for item in labels:
        zero_list = list([0] * 10)
        zero_list[item] = 1
        labels_list.append(zero_list)
    y_train = np.array(labels_list)
    return np.array(images), y_train

def run(train_images, train_labels, test_images, test_labels):
    X_train, y_train = pre_process(train_images, train_labels)
    X_test, y_test = pre_process(test_images, test_labels)
    w1, b1, w2, b2, w3, b3, w4, b4 = train(X_train, y_train, X_train, train_labels, X_test, test_labels)
    '''
    np.save("./mnist_relu_network_weights/w1.npy", w1)
    np.save("./mnist_relu_network_weights/b1.npy", b1)
    np.save("./mnist_relu_network_weights/w2.npy", w2)
    np.save("./mnist_relu_network_weights/b2.npy", b2)
    np.save("./mnist_relu_network_weights/w3.npy", w3)
    np.save("./mnist_relu_network_weights/b3.npy", b3)
    np.save("./mnist_relu_network_weights/w4.npy", w4)
    np.save("./mnist_relu_network_weights/b4.npy", b4)
    '''
    test_accuracy = evaluate(X_test, test_labels, w1, b1, w2, b2, w3, b3, w4, b4)
    train_accuracy = evaluate(X_train, train_labels,  w1, b1, w2, b2, w3, b3, w4, b4)
    print("Accuracy of train_set: {0}".format(train_accuracy))
    print("Accuracy of test_set: {0}".format(test_accuracy))
    record(train_accuracy, test_accuracy)

def load_data():
    trainset = fetch_trainset()
    testset = fetch_testset()
    X_train, y_train = trainset["images"], trainset["labels"]
    X_test, y_test = testset["images"], testset["labels"]
    return X_train, y_train, X_test, y_test

def record(train_accuracy, test_accuracy):
    f = open("./record_relu.txt", "a")
    f.write("\nLAYER1_NODE = {0}, LAYER2_NODE = {1}, LAYER3_NODE = {2} BATCH_SIZE = {3}, learning_rate = {4}, TRAINING_STEPS = {5}, WEIGHT_BASE = {6}".format(LAYER1_NODE, \
            LAYER2_NODE, LAYER3_NODE, BATCH_SIZE, learning_rate, TRAINING_STEPS, WEIGHT_BASE))
    f.write("\nAccuracy of train_set: " + str(train_accuracy))
    f.write("\nAccuracy of test_set: " + str(test_accuracy))
    f.close()