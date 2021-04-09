import numpy as np
import math
import json
import os
import shutil
import csv

from models import Neuron
from activation_funcs import sign_activation, tanh_activation, lineal_activation

weights_folder_path = 'weights'
ej1_model_filename = 'ej1_model.json'

ej2_training = "data/ej2-Conjuntoentrenamiento.txt"
ej2_outputs = "data/ej2-Salida-deseada.txt"

LIMIT = 5000000

def initialize_simple_perceptron(activation_func):
    initial_weights = np.random.uniform(-1, 1, X.shape[1])
    initial_bias = np.random.uniform(-1, 1)
    print(f"initial values: weights={initial_weights} / bias={initial_bias}")
    return Neuron(initial_weights, initial_bias, activation_func)

def calculate_error(perceptron, X, y):
    err = 0
    for i in range(0, X.shape[0]):
        #print(f'expected: {y[i]}, output: {perceptron.evaluate(X[i, :])}')
        err += abs(perceptron.evaluate(X[i, :]) - y[i])
    return err

def calculate_mean_error(perceptron, X, y):
    err = 0
    for i in range(0, X.shape[0]):
        #print(f'expected: {y[i]}, output: {perceptron.evaluate(X[i, :])}')
        err += abs(perceptron.evaluate(X[i, :]) - y[i])
    err /= X.shape[0]
    return err

def calculate_standard_deviation(perceptron, X, y):
    stdev = 0
    for i in range(0, X.shape[0]):
        #print(f'expected: {y[i]}, output: {perceptron.evaluate(X[i, :])}')
        stdev += pow(perceptron.evaluate(X[i, :]) - y[i], 2)
    stdev /= X.shape[0]
    stdev = math.sqrt(stdev)
    return stdev

def parse_data(file):
    datafile = open(file, 'r')
    datareader = csv.reader(datafile, delimiter=' ')
    data = []
    for row in datareader:
        clean_row = [float(a) for a in row if a != '']
        if len(clean_row) == 1:
            data.append(clean_row[0]) 
        else:
            data.append(clean_row)   
    return np.array(data)

if __name__ == "__main__":

    weights = {}
    learning_level = 0.03
    
     #EJ 1 ------------------------------------------------

    # and
    # X = np.array([
    #     [-1, 1],
    #     [1, -1],
    #     [-1, -1],
    #     [1, 1]
    # ])

    # y = np.array([-1, -1, -1, 1])
    
    # xor (no se puede)
    # X = np.array([
    #     [-1, 1],
    #     [1, -1],
    #     [-1, -1],
    #     [1, 1]
    # ])

    # y = np.array([1, 1, -1, -1])


    #EJ 2 ------------------------------------------------
    X = parse_data(ej2_training)
    y = parse_data(ej2_outputs)
    
    #initialize perceptron
    simple_perceptron = initialize_simple_perceptron(lineal_activation)
    w_min = simple_perceptron.weights 
    b_min = simple_perceptron.bias

    #train perceptron
    epsilon = .2
    i = 0
    n = 0
    error_min = math.inf # X.shape[0] * 2
    error = error_min
    while error > epsilon and i < LIMIT:
        
        #check if perceptron needs reseting
        # if n > 100000 * X.shape[0]:
        #     simple_perceptron = initialize_simple_perceptron(lineal_activation)
        #     n = 0
        #     print("Reseting perceptron")

        #choose random input            
        rand_idx = np.random.randint(0, X.shape[0])
        #print(f'rand_idx: {rand_idx}')
        #print("X:", X)
        rand_X = X[rand_idx, :]
        rand_y = y[rand_idx]

        #evaluate chocen input
        activation = simple_perceptron.evaluate(rand_X)
        simple_perceptron.apply_correction(learning_level, rand_y, activation, rand_X)

        #print(f'weights: {simple_perceptron.weights}')
        
        #calculate training error
        #if(i%10 == 0):
        error = calculate_error(simple_perceptron, X, y)
        #print(f'error: {error}')
        
        if error < error_min:
            error_min = error
            w_min = simple_perceptron.weights.copy() 
            b_min = simple_perceptron.bias

            print('updated error_min', error_min)
            #print('w_min:', w_min)
            #print('b_min:', b_min)

            #for i in range(0, X.shape[0]):
            #    print(f'X = {X[i,:]} => y = {simple_perceptron.evaluate(X[i,:])} (should return {y[i]})')

        #print(f'w_min: {w_min}')

        i += 1
        n += 1

    if i >= LIMIT:
        print("------------> LIMIT passed")
        
    print("Finished")

    best_perceptron = Neuron(w_min, b_min, lineal_activation)

    print(f"standard_deviation: {calculate_standard_deviation(best_perceptron, X, y)}")
    print(f"error_mean: {calculate_mean_error(best_perceptron, X, y)}")
    print(f"error_min: {error_min}")
    print(f'w_min: {w_min}')
    print(f'b_min: {b_min}')

    for i in range(0, X.shape[0]):
        print(f'X = {X[i,:]} => y = {best_perceptron.evaluate(X[i,:])} (should return {y[i]})')

    if error_min == 0:
        if not (os.path.exists(weights_folder_path) and os.path.isdir(weights_folder_path)):
            os.mkdir(weights_folder_path)

        def to_float(e):
            return float(e)

        w_min_l = w_min.tolist()
        map(to_float, w_min_l)

        results = {
            'and': {
                'weights': w_min_l,
                'bias': float(b_min),
                'training': {
                    'error': float(error_min)
                }
            }
            # # Arreglar
            # 'xor': {
            #     'weights': w_min,
            #     'bias': b_min,
            #     'training': {
            #         'error': error_min
            #     }
            # }
        }

        print(f'Results: {results}')

        filename = os.path.join(weights_folder_path, ej1_model_filename)

        with open(filename, 'w') as outfile:
            json.dump(results, outfile)
            print(f"Model saved: '{filename}'")

        
    print('Weights:', w_min)
    print('Bias:', b_min)