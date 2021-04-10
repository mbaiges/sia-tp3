import numpy as np
import math
import json
import os
import shutil

from models import Neuron, SimplePerceptron
from utils import import_and_parse_data, calculate_abs_error, calculate_mean_error, calculate_standard_deviation
from activation_funcs import sign_activation, tanh_activation, lineal_activation

weights_folder_path = 'weights'
ej1_model_filename = 'ej1_model.json'

ej2_training = "data/ej2-Conjuntoentrenamiento.txt"
ej2_outputs = "data/ej2-Salida-deseada.txt"

LIMIT = 5000000

if __name__ == "__main__":

    weights = {}
    learning_level = 0.03
    
    #EJ 1 ------------------------------------------------

    # and
    X = np.array([
        [-1, 1],
        [1, -1],
        [-1, -1],
        [1, 1]
    ])

    y = np.array([-1, -1, -1, 1])
    
    # xor (no se puede)
    # X = np.array([
    #     [-1, 1],
    #     [1, -1],
    #     [-1, -1],
    #     [1, 1]
    # ])

    # y = np.array([1, 1, -1, -1])


    #EJ 2 ------------------------------------------------
    # X = import_and_parse_data(ej2_training)
    # y = import_and_parse_data(ej2_outputs)
    
    #initialize perceptron
    simple_perceptron = SimplePerceptron(X.shape[1], learning_level, sign_activation)

    #train perceptron
    result = simple_perceptron.train(X, y)

    if result:
        print("------------> LIMIT passed")
        
    print("Finished")

    best_neuron = simple_perceptron.best_neuron()

    abs_error = simple_perceptron.min_error
    mean_error = calculate_mean_error(best_neuron, X, y)
    standard_deviation = calculate_standard_deviation(best_neuron, X, y)

    print(f"min_error: {abs_error}")
    print(f"mean_error: {mean_error}")
    print(f"standard_deviation: {standard_deviation}")

    print(f'min_weights: {simple_perceptron.min_weights}')
    print(f'min_bias: {simple_perceptron.min_bias}')

    best_neuron.print_predictions_with_expected(X, y)

    if not (os.path.exists(weights_folder_path) and os.path.isdir(weights_folder_path)):
        os.mkdir(weights_folder_path)

    def to_float(e):
        return float(e)

    min_weights_l = best_neuron.weights.tolist()
    map(to_float, min_weights_l)

    results = {
        'and': {
            'weights': min_weights_l,
            'bias': float(best_neuron.bias),
            'training': {
                'abs_error': float(abs_error),
                'mean_error': float(mean_error),
                'standard_deviation': float(standard_deviation),
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

        
    print('Weights:', best_neuron.weights)
    print('Bias:', best_neuron.bias)