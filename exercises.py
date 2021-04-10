import numpy as np
import json
import os
import shutil

from models import Neuron, SimplePerceptron
from utils import import_and_parse_data, calculate_abs_error, calculate_mean_error, calculate_standard_deviation, print_predictions_with_expected
from activation_funcs import sign_activation, lineal_activation, tanh_activation

saves_folder_path = 'saves'

# Ej1

## results filenames

ej1_and_filename = 'ej1_and.json'
ej1_xor_filename = 'ej1_xor.json'

# Ej2

## results filenames

ej2_lineal_filename = 'ej2_lineal.json'
ej2_no_lineal_filename = 'ej2_no_lineal.json'
ej2_no_lineal_with_testing_filename = 'ej2_no_lineal_with_testing.json'

## data filenames

ej2_training = "data/ej2-Conjuntoentrenamiento.txt"
ej2_outputs = "data/ej2-Salida-deseada.txt"

class Exercise:

    def train(self):
        pass

    def predict(self):
        pass

class SimplePerceptronExerciseTemplate(Exercise):

    def __init__(self, activation_func, epsilon, limit, max_it_same_bias, training_level, filename):
        self.activation_func = activation_func
        self.epsilon = epsilon
        self.limit = limit
        self.max_it_same_bias = max_it_same_bias
        self.training_level = training_level
        self.filename = filename

    def get_data(self):
        pass

    def get_analysis_results(self, perceptron, X, y):
        best_neuron = perceptron.best_neuron()

        abs_error = perceptron.min_error
        mean_error = calculate_mean_error(best_neuron, X, y)
        standard_deviation = calculate_standard_deviation(best_neuron, X, y)

        results = {
            'abs_error': float(abs_error),
            'mean_error': float(mean_error),
            'standard_deviation': float(standard_deviation),
        }

        return results

    def build_results(self, perceptron, training_results, testing_results):
        best_neuron = perceptron.best_neuron()  
    
        def to_float(e):
            return float(e)

        min_weights_l = best_neuron.weights.tolist()
        map(to_float, min_weights_l)

        results = {
            'weights': min_weights_l,
            'bias': float(best_neuron.bias),
            'training': training_results
        }

        if not testing_results is None:
            results['testing'] = testing_results

        return results

    def save_results(self, results):
        if not (os.path.exists(saves_folder_path) and os.path.isdir(saves_folder_path)):
            os.mkdir(saves_folder_path)

        filename = os.path.join(saves_folder_path, self.filename)

        with open(filename, 'w') as outfile:
            json.dump(results, outfile)
            print(f"Model saved: '{filename}'")

    def read_last_results(self):
        if not (os.path.exists(saves_folder_path) and os.path.isdir(saves_folder_path)):
            print(f"Error: folder \'{saves_folder_path}\' does not exist")
            exit(1)

        filename = os.path.join(saves_folder_path, self.filename)

        if not (os.path.exists(filename) and os.path.isfile(filename)):
            print(f"Error: file \'{filename}\' does not exist")
            exit(1)

        with open(filename) as json_file:
            return json.load(json_file)

    # get new weights with a new training
    def train_and_test(self):
        X_train, y_train, X_test, y_test = self.get_data()

        # initialize perceptron
        simple_perceptron = SimplePerceptron(X_train.shape[1], self.training_level, self.activation_func)

        # train perceptron
        result = simple_perceptron.train(X_train, y_train)

        if result:
            print(f"LIMIT ({self.limit}) passed")
            
        print("Finished")

        training_results = self.get_analysis_results(simple_perceptron, X_train, y_train)

        # test perceptron
        testing_results = None

        if X_test.shape[0] > 0:
            testing_results = self.get_analysis_results(simple_perceptron, X_test, y_test)

        results = self.build_results(simple_perceptron, training_results, testing_results)

        results_printing = json.dumps(results, sort_keys=False, indent=4, default=str)
        print(results_printing)

        self.save_results(results)

    # predict output based on saved model
    def predict(self):
        # model = return
        pass

class Ej1And(SimplePerceptronExerciseTemplate):

    def __init__(self):
        super().__init__(sign_activation, 0, 100000, 10000, 0.01, ej1_and_filename)

    def get_data(self):
        X = np.array([
            [-1, 1],
            [1, -1],
            [-1, -1],
            [1, 1]
        ])

        y = np.array([-1, -1, -1, 1])

        return X, y, np.array([]), np.array([])

class Ej1Xor(SimplePerceptronExerciseTemplate):

    def __init__(self):
        super().__init__(sign_activation, 100000, 10000, 0.01, ej1_xor_filename)

    def get_data(self):
        X = np.array([
            [-1, 1],
            [1, -1],
            [-1, -1],
            [1, 1]
        ])

        y = np.array([1, 1, -1, -1])

        return X, y, np.array([]), np.array([])

    def train_and_test(self):
        print("Warning: XOR is not linearly separable")
        super().train_and_test()

class Ej2(SimplePerceptronExerciseTemplate):

    def get_data(self):
        X = import_and_parse_data(ej2_training)
        y = import_and_parse_data(ej2_outputs)

        return X, y, np.array([]), np.array([])

class Ej2Lineal(Ej2):

    def __init__(self):
        super().__init__(lineal_activation, .5, 100000, 10000, 0.03, ej2_lineal_filename)

class Ej2NoLineal(Ej2):

    def __init__(self):
        super().__init__(tanh_activation, .5, 100000, 10000, 0.03, ej2_no_lineal_filename)
    
class Ej2NoLinealWithTesting(Ej2):

    def __init__(self):
        super().__init__(tanh_activation, .5, 100000, 10000, 0.03, ej2_no_lineal_with_testing_filename)

    def get_data(self):
        X_train, y_train, _, _ = super().get_data()

        return X_train, y_train, np.array([]), np.array([])
