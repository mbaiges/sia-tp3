import numpy as np
import math
import random
import yaml
import os
import multiprocessing as mp
import keyboard

from utils import calculate_abs_error, calculate_mean_error
from plotters import plot_avg_error

plotting = False

config_filename = 'config.yaml'

# extra modes

momentum = False
momentum_alpha = 0.8

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    if 'plotting' in config:
        plotting = config['plotting']

    if 'momentum' in config and 'opt' in config['momentum'] and 'alpha' in config['momentum']:
        momentum = config['momentum']['opt']
        momentum_alpha = config['momentum']['alpha']

class SimplePerceptron:

    def __init__(self, X_shape, learning_level, activation_func):
        self.learning_level = learning_level
        self.activation_func = activation_func
        self.X_shape = X_shape
        self.initialize()
        self.min_weights_and_bias = self.neuron.get_weights_and_bias()
        self.min_error = math.inf

    def initialize(self):
        initial_weights = np.random.uniform(-1, 1, self.X_shape)
        initial_bias = np.random.uniform(-1, 1)
        self.neuron = Neuron(initial_weights, initial_bias, self.activation_func)

    def train(self, X, y, epsilon=0, epochs=100, max_it_same_bias=1000):
        
        error = self.min_error
        
        #creat e random= index array 
        orders = [a for a in range(0, X.shape[0])]
        
        epoch_n = 0

        if plotting:
            plotter_q = mp.Queue()
            plotter_q.cancel_join_thread()

            plotter = mp.Process(target=plot_avg_error, args=((plotter_q),))
            plotter.daemon = True
            plotter.start()

        while epoch_n < epochs and error > epsilon:
            random.shuffle(orders)
            
            i = 0
            n = 0
            
            while i < len(orders) and error > epsilon:

                #check if perceptron needs resetting
                if n > max_it_same_bias * X.shape[0]:
                    self.initialize()
                    n = 0

                #choose random input            
                # rand_idx = np.random.randint(0, X.shape[0])
                # rand_X = X[rand_idx, :]
                # rand_y = y[rand_idx]
                
                #access index from order array
                indx = orders[i]
                pos_X = X[indx, :]
                pos_y = y[indx]

                #evaluate chosen input
                activation = self.neuron.evaluate(pos_X)
                self.neuron.apply_correction(self.learning_level, pos_y, activation, pos_X)
                
                #calculate training error
                error = calculate_abs_error(self.neuron, X, y)

                if plotting:
                    mean_error = calculate_mean_error(self.neuron, X, y)
                    plotter_q.put({
                        'mean_error': mean_error
                    })
                
                if error < self.min_error:
                    self.min_error = error
                    self.min_weights_and_bias = self.neuron.get_weights_and_bias()
                    #print('updated min_error', self.min_error)

                i += 1
                n += 1

            epoch_n += 1

        if plotting:
            plotter_q.put("STOP")
            print("Press 'q' to finish plot")
            keyboard.wait("q")

        return epoch_n >= epochs

    def get_best_model(self):
        return Neuron(self.min_weights_and_bias['weights'], self.min_weights_and_bias['bias'], self.activation_func)

class MultilayerPerceptron:

    def __init__(self, hidden_layers, X_shape, Y_shape, learning_level, activation_func, dx_activation_func):
        self.hidden_layers = hidden_layers
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.learning_level = learning_level
        self.activation_func = activation_func
        self.dx_activation_func = dx_activation_func
        self.initialize()
        self.min_weights_and_bias = self.neural_network.get_weights_and_bias()
        self.min_error = math.inf

    def initialize(self):
        self.neural_network = NeuralNetwork(self.hidden_layers, self.X_shape, self.Y_shape, self.activation_func, self.dx_activation_func)

    def train(self, X, Y, epsilon=0, epochs=100, max_it_same_bias=1000):
        i = 0
        n = 0
        error = self.min_error

        #creat e random= index array 
        orders = [a for a in range(0, X.shape[0])]
        
        epoch_n = 0

        if plotting:
            plotter_q = mp.Queue()
            plotter_q.cancel_join_thread()

            plotter = mp.Process(target=plot_avg_error, args=((plotter_q),))
            plotter.daemon = True
            plotter.start()

        while epoch_n < epochs and error > epsilon:
            random.shuffle(orders)
            
            i = 0
            n = 0
            
            while i < len(orders) and error > epsilon:
            
                #check if perceptron needs resetting
                if n > max_it_same_bias * X.shape[0]:
                    self.initialize()
                    n = 0

                #choose random input            
                rand_idx = np.random.randint(0, X.shape[0])
                rand_X = X[rand_idx, :]
                rand_Y = Y[rand_idx]

                #print("holaaa1")
                #evaluate chosen input
                activation = self.neural_network.evaluate(rand_X)
                self.neural_network.apply_correction(self.learning_level, rand_Y, activation, rand_X)
                # print("holaaa2")
                #calculate training error
                error = calculate_abs_error(self.neural_network, X, Y)

                if plotting:
                    mean_error = calculate_mean_error(self.neural_network, X, Y)
                    plotter_q.put({
                        'mean_error': mean_error
                    })

                #print(f"Current error: {error}")
                
                if error < self.min_error:
                    self.min_error = error
                    self.min_weights_and_bias = self.neural_network.get_weights_and_bias()
                    #print('updated min_error', self.min_error)

                i += 1
                n += 1

            epoch_n += 1

        if plotting:
            plotter_q.put("STOP")
            print("Press 'q' to finish plot")
            keyboard.wait("q")

        return epoch_n >= epochs

    def get_best_model(self):
        return NeuralNetwork(self.min_weights_and_bias, self.hidden_layers, self.X_shape, self.Y_shape, self.activation_func, self.dx_activation_func)

class Neuron:

    def __init__(self, weights, bias, activation_func):
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func
        
        if momentum:
            self.last_delta_weights = None
            self.last_delta_bias = None

    def apply_correction(self, learning_level, expected_result, result, entry):
        correction = learning_level * (expected_result - result)
        #print(f'Correction: {correction} [{expected_result}/{result}]')
        delta_weights = correction * entry
        #print(f'delta_weights: {delta_weights}')
        delta_bias = correction
        #print(f'Old weights: {self.weights}, bias: {self.bias}')
        
        if momentum:
            if not self.last_delta_weights is None and not self.last_delta_bias is None:
                delta_weights = delta_weights + momentum_alpha * self.last_delta_weights
                delta_bias = delta_bias + momentum_alpha * self.last_delta_bias

            self.last_delta_weights = delta_weights
            self.last_delta_bias = delta_bias
        
        self.weights += delta_weights
        self.bias += delta_bias
        #print(f'New weights: {self.weights}, bias: {self.bias}')

    def evaluate(self, entry):
        #print('Evaluating:', entry)
        #print('Weights:', self.weights)
        excitation = np.inner(entry, self.weights)
        #print(f"\tExcitation: {excitation}")
        activation = self.activation_func(excitation + self.bias)
        #print(f"\tActivation: {activation}")
        return activation

    def get_weights_and_bias(self):
        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }

class NetworkNeuron:

    def __init__(self, weights, bias, activation_func):
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func
        
    def apply_correction(self, learning_level, entry, delta): 
        self.last_delta = delta
        correction = learning_level * delta
        delta_weights = correction * entry
        delta_bias = correction
        self.weights += delta_weights
        self.bias += delta_bias
        
    def evaluate(self, entry):
        excitation = np.inner(entry, self.weights)
        self.last_excitation = excitation
        activation = self.activation_func(excitation + self.bias)
        self.last_activation = activation
        return activation


class NeuralNetwork:

    def __init__(self, *args):

        if len(args) == 5:
            self.__init__1(args[0], args[1], args[2], args[3], args[4])
        else:
            self.__init__2(args[0], args[1], args[2], args[3], args[4], args[5])

    def __init__1(self, hidden_layers, X_shape, Y_shape, activation_func, dx_activation_func):
        # print(f'Hidden layers: {hidden_layers}')
        # print(X_shape)
        self.hidden_layers = hidden_layers
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.activation_func = activation_func
        self.dx_activation_func = dx_activation_func
        self.network = self.create_network()

    def __init__2(self, weights_and_bias, hidden_layers, X_shape, Y_shape, activation_func, dx_activation_func):
        self.hidden_layers = hidden_layers
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.activation_func = activation_func
        self.dx_activation_func = dx_activation_func
        self.network = self.create_network_with_weights_and_bias(weights_and_bias)

    def create_network_with_weights_and_bias(self, weights_and_bias):

        net = []
        
        for i in range(0, len(self.hidden_layers)):
            net.append([])
            for j in range(0, self.hidden_layers[i]):
                weights_len = self.X_shape
                if i > 0:
                    weights_len = self.hidden_layers[i-1]
                initial_weights = weights_and_bias[i][j]['weights']
                initial_bias = weights_and_bias[i][j]['bias']
                new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
                net[i].append(new_neuron)

        last_layer = []

        for k in range(0, self.Y_shape):
            weights_len = self.hidden_layers[-1]
            initial_weights = weights_and_bias[-1][k]['weights']
            initial_bias = weights_and_bias[-1][k]['bias']
            new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
            last_layer.append(new_neuron)

        net.append(last_layer)

        return net

    def create_network(self):

        net = []
        
        for i in range(0, len(self.hidden_layers)):
            net.append([])
            for j in range(0, self.hidden_layers[i]):
                weights_len = self.X_shape
                if i > 0:
                    weights_len = self.hidden_layers[i-1]
                initial_weights = np.random.uniform(-1, 1, weights_len)
                initial_bias = np.random.uniform(-1, 1)
                new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
                net[i].append(new_neuron)

        last_layer = []

        for k in range(0, self.Y_shape):
            weights_len = self.hidden_layers[-1]
            initial_weights = np.random.uniform(-1, 1, weights_len)
            initial_bias = np.random.uniform(-1, 1)
            new_neuron = NetworkNeuron(initial_weights, initial_bias, self.activation_func)
            last_layer.append(new_neuron)

        net.append(last_layer)

        return net
            
    # backpropagation
    def apply_correction(self, learning_level, expected_result, result, entry):
        for l in range(0, len(self.network)):
            i = len(self.network)-1 - l
            if i == 0:
                entries = entry
            else:
                entries = np.array([n.last_activation for n in self.network[i-1]])

            for j in range(0, len(self.network[i])):
                neuron = self.network[i][j]
                if i == len(self.network) - 1:
                    delta = (expected_result - result) * self.dx_activation_func(neuron.last_excitation)
                else:
                    result = 0
                    for k in range(0, len(self.network[i+1])):
                        parent_neuron = self.network[i+1][k]
                        result += parent_neuron.weights[j] * parent_neuron.last_delta
                    delta = self.dx_activation_func(neuron.last_excitation) * result

                
                self.network[i][j].apply_correction(learning_level, entries, delta)
        
    def evaluate(self, entry):
        entries = entry
        for i in range(0, len(self.network)):
            if i > 0:
                entries = np.array([n.last_activation for n in self.network[i-1]])
            for j in range(0, len(self.network[i])):
                neuron = self.network[i][j]
                neuron.evaluate(entries)

        result = np.array([n.last_activation for n in self.network[-1]])

        if result.shape[0] == 1:
            result = result[0]
        
        return result

    def get_weights_and_bias(self):

        wab = []

        for i in range(0, len(self.network)):
            wab.append([])
            for j in range(0, len(self.network[i])):
                neuron = self.network[i][j]
                wab[i].append({
                    'weights': neuron.weights.copy(),
                    'bias': neuron.bias
                })

        return wab