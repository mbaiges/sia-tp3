import math
import numpy as np
import json
import os
import shutil
import ast
import re
import yaml

from models import Neuron, NeuralNetwork, SimplePerceptron, MultilayerPerceptron
from utils import import_and_parse_data, import_and_parse_numbers, calculate_abs_error, calculate_mean_error, calculate_standard_deviation, print_predictions_with_expected
from activation_funcs import sign_activation, lineal_activation, tanh_activation, dx_sign_activation, dx_lineal_activation, dx_tanh_activation

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

ej2_training = ""
ej2_outputs = ""

# Ej3

## results filenames

ej3_xor_filename = 'ej3_xor.json'
ej3_pair_filename = 'ej3_pair.json'

## data filenames

ej3_pair_training = ""

config_filename = 'config.yaml'

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    data_folder = config['data_folder']

    ej2_training = os.path.join(data_folder, config['ej2_training'])
    ej2_outputs = os.path.join(data_folder, config['ej2_outputs'])

    ej3_pair_training = os.path.join(data_folder, config['ej3_pair_training'])

class Exercise:

    def train_and_test(self):
        pass

    def predict(self):
        pass
    
class PerceptronExerciseTemplate(Exercise):

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
        best_model = perceptron.get_best_model()

        abs_error = calculate_abs_error(best_model, X, y)
        mean_error = calculate_mean_error(best_model, X, y)
        standard_deviation = calculate_standard_deviation(best_model, X, y)

        results = {
            'abs_error': float(abs_error),
            'mean_error': float(mean_error),
            'standard_deviation': float(standard_deviation),
        }

        return results

    def build_results(self, perceptron, training_results, testing_results):
        best_model = perceptron.get_best_model()  

        weights_and_bias = best_model.get_weights_and_bias()

        if isinstance(weights_and_bias, list):
            for i in range(0, len(weights_and_bias)):
                for j in range(0, len(weights_and_bias[i])):
                    weights_and_bias[i][j]['weights'] = weights_and_bias[i][j]['weights'].tolist()
                    map(lambda e: float(e), weights_and_bias[i][j]['weights'])
                    weights_and_bias[i][j]['bias'] = float(weights_and_bias[i][j]['bias'])
        else:
            weights_and_bias['weights'] = weights_and_bias['weights'].tolist()
            map(lambda e: float(e), weights_and_bias['weights'])
            weights_and_bias['bias'] = float(weights_and_bias['bias'])

        results = {
            'configuration': weights_and_bias,
            'training': {
                'params': {
                    'epsilon': self.epsilon,
                    'limit': self.limit,
                    'max_it_same_bias': self.max_it_same_bias,
                    'training_level': self.training_level
                },
                'analysis': training_results
            }
        }

        if not testing_results is None:
            results['testing'] = { 'analysis': testing_results }

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

    def interactive_predict(self, model, X_shape):
        predicted = False
        wants_to_keep_predicting = True

        while not predicted or wants_to_keep_predicting:

            selected_X = None
            error = False

            while wants_to_keep_predicting and (selected_X is None or error or len(selected_X) != X_shape):
                if error:
                    error = False
                    print("Error: Invalid input")
                if not (selected_X is None or error) and (selected_X) != X_shape:
                    selected_X = None
                    print(f"Error: Invalid X shape (expected: {X_shape})")

                inp = input("Select an entry (Example: [ 0.2, 0.3 ]): ").strip()
                wants_to_keep_predicting = False if inp == '' else True

                if wants_to_keep_predicting:
                    if re.match('^\[[ \t]*((\.[0-9]+)|([0-9]+(\.[0-9]*)?))[ \t]*(,[ \t]*((\.[0-9]+)|([0-9]+(\.[0-9]*)?))[ \t]*)*\]$', inp):
                        try:
                            selected_X = ast.literal_eval(inp)
                            error = False if isinstance(selected_X, list) else True
                        except ValueError:
                            error = True
                    else:
                        error = True

            if wants_to_keep_predicting:
                print(f"Evaluating {selected_X}")
                print(f"Returned {model.evaluate(selected_X)}")
            
            predicted = True

    # predict output based on saved model
    def predict(self):
        # model = return
        pass

class SimplePerceptronExerciseTemplate(PerceptronExerciseTemplate):

    # get new weights with a new training
    def train_and_test(self):
        X_train, y_train, X_test, y_test = self.get_data()

        # initialize perceptron
        simple_perceptron = SimplePerceptron(X_train.shape[1], self.training_level, self.activation_func)

        # train perceptron
        print("Started training")
        result = simple_perceptron.train(X_train, y_train, self.epsilon, self.limit, self.max_it_same_bias)

        if result:
            print(f"LIMIT ({self.limit}) passed")
            
        print("Finished training")

        training_results = self.get_analysis_results(simple_perceptron, X_train, y_train)

        # test perceptron
        testing_results = None

        if X_test.shape[0] > 0:
            print("Started testing")
            testing_results = self.get_analysis_results(simple_perceptron, X_test, y_test)
            print("Finished testing")

        results = self.build_results(simple_perceptron, training_results, testing_results)

        results_printing = json.dumps(results, sort_keys=False, indent=4, default=str)
        print(results_printing)

        self.save_results(results)

    def predict(self):
        last_results = self.read_last_results()
        configuration = last_results['configuration']

        neuron = Neuron(configuration['weights'], configuration['bias'], self.activation_func)

        X_shape = len(configuration['weights'])

        self.interactive_predict(neuron, X_shape)


class MultilayerPerceptronExerciseTemplate(PerceptronExerciseTemplate):

    def __init__(self, hidden_layers, activation_func, dx_activation_func, epsilon, limit, max_it_same_bias, training_level, filename):
        super().__init__(activation_func, epsilon, limit, max_it_same_bias, training_level, filename)
        self.dx_activation_func = dx_activation_func
        self.hidden_layers = hidden_layers # las layers vienen en la forma [a, b, c, d...] donde cada letra representa las neuronas de cada capa

    # get new weights with a new training
    def train_and_test(self):
        X_train, Y_train, X_test, Y_test = self.get_data()

        # initialize perceptron
        Y_shape = 1 if len(Y_train.shape) == 1 else Y_train.shape[1]
        neural_net = MultilayerPerceptron(self.hidden_layers, X_train.shape[1], Y_shape, self.training_level, self.activation_func, self.dx_activation_func)
       
        # train perceptron
        print("Started training")
        result = neural_net.train(X_train, Y_train, self.epsilon, self.limit, self.max_it_same_bias)

        if result:
            print(f"LIMIT ({self.limit}) passed")
            
        print("Finished training")

        training_results = self.get_analysis_results(neural_net, X_train, Y_train)

        # test perceptron
        testing_results = None

        if X_test.shape[0] > 0:
            print("Started testing")
            testing_results = self.get_analysis_results(neural_net, X_test, Y_test)
            print("Finished testing")

        results = self.build_results(neural_net, training_results, testing_results)

        results_printing = json.dumps(results, sort_keys=False, indent=4, default=str)
        print(results_printing)

        self.save_results(results)

    def predict(self):
        last_results = self.read_last_results()

        configuration = last_results['configuration']

        
        X_shape = len(configuration[0][0]['weights'])
        Y_shape = len(configuration[-1])

        network = NeuralNetwork(configuration, self.hidden_layers, X_shape, Y_shape, self.activation_func, self.dx_activation_func)

        self.interactive_predict(network, X_shape)


class Ej1And(SimplePerceptronExerciseTemplate):

    def __init__(self):
        epsilon = 0
        limit = 100000
        max_it_same_bias = 10000
        training_level = 0.01
        super().__init__(sign_activation, epsilon, limit, max_it_same_bias, training_level, ej1_and_filename)

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
        epsilon = 0
        limit = 100000
        max_it_same_bias = 10000
        training_level = 0.01
        super().__init__(sign_activation, epsilon, limit, max_it_same_bias, training_level, ej1_xor_filename)

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

        y_max = np.amax(y)

        print(f"Max output: {y_max}")
        print(f"Dividing all outputs by {y_max}")

        y /= y_max
        y = y * 2 - 1

        return X, y, np.array([]), np.array([])

class Ej2Lineal(Ej2):

    def __init__(self):
        epsilon = .001
        limit = 10000
        max_it_same_bias = 10000
        training_level = 0.01
        super().__init__(lineal_activation, epsilon, limit, max_it_same_bias, training_level, ej2_lineal_filename)

class Ej2NoLineal(Ej2):

    def __init__(self):
        epsilon = .001
        limit = 10000
        max_it_same_bias = 10000
        training_level = 0.01
        super().__init__(tanh_activation, epsilon, limit, max_it_same_bias, training_level, ej2_no_lineal_filename)
    
class Ej2NoLinealWithTesting(Ej2):

    def __init__(self):
        epsilon = .001
        limit = 10000
        max_it_same_bias = 10000
        training_level = 0.03
        super().__init__(tanh_activation, epsilon, limit, max_it_same_bias, training_level, ej2_no_lineal_with_testing_filename)
        self.training_pctg = .7

    def get_data(self):
        X, y, _, _ = super().get_data()

        
        break_point = int(math.ceil(X.shape[0] * self.training_pctg))

        X_train = X[0:break_point]
        y_train = y[0:break_point]

        X_test = X[break_point:-1]
        y_test = y[break_point:-1]

        return X_train, y_train, X_test, y_test

    def build_results(self, perceptron, training_results, testing_results):
        results = super().build_results(perceptron, training_results, testing_results)

        results['training_pctg'] = self.training_pctg

        return results

class Ej3Xor(MultilayerPerceptronExerciseTemplate):

    def __init__(self):
        epsilon = .001
        limit = 10000
        max_it_same_bias = 10000
        training_level = 0.01
        super().__init__([3, 3, 3], tanh_activation, dx_tanh_activation, epsilon, limit, max_it_same_bias, training_level, ej3_xor_filename)

    def get_data(self):
        X = np.array([
            [-1, 1],
            [1, -1],
            [-1, -1],
            [1, 1]
        ])

        y = np.array([1, 1, -1, -1])

        return X, y, np.array([]), np.array([])

class Ej3Pair(MultilayerPerceptronExerciseTemplate):

    def __init__(self):
        epsilon = .001
        limit = 10000
        max_it_same_bias = 10000
        training_level = 0.01
        self.training_pctg = .5
        super().__init__([3, 3, 3], tanh_activation, dx_tanh_activation, epsilon, limit, max_it_same_bias, training_level, ej3_pair_filename)

    def get_data(self):
        X = import_and_parse_numbers(ej3_pair_training)
        y = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])

        print("=========>>>>", round(self.training_pctg, 1))
        break_point = int(math.ceil(X.shape[0] * round(self.training_pctg, 1)))

        X_train = X[0:break_point]
        y_train = y[0:break_point]

        X_test = X[break_point:-1]
        y_test = y[break_point:-1]

        return X_train, y_train, X_test, y_test

    def build_results(self, perceptron, training_results, testing_results):
        results = super().build_results(perceptron, training_results, testing_results)

        results['training_pctg'] = self.training_pctg

        return results