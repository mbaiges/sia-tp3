import numpy as np
import math

from utils import calculate_abs_error

class SimplePerceptron:

    def __init__(self, X_shape, learning_level, activation_func):
        self.learning_level = learning_level
        self.activation_func = activation_func
        self.X_shape = X_shape
        self.initialize()
        self.min_weights = self.neuron.weights 
        self.min_bias = self.neuron.bias
        self.min_error = math.inf # X.shape[0] * 2

    def initialize(self):
        initial_weights = np.random.uniform(-1, 1, self.X_shape)
        initial_bias = np.random.uniform(-1, 1)
        self.neuron = Neuron(initial_weights, initial_bias, self.activation_func)

    def train(self, X, y, epsilon=0, limit=100000, max_it_same_bias=1000):
        i = 0
        n = 0
        error = self.min_error
        while error > epsilon and i < limit:
            
            #check if perceptron needs resetting
            if n > max_it_same_bias * X.shape[0]:
                self.initialize()
                n = 0

            #choose random input            
            rand_idx = np.random.randint(0, X.shape[0])
            rand_X = X[rand_idx, :]
            rand_y = y[rand_idx]

            #evaluate chosen input
            activation = self.neuron.evaluate(rand_X)
            self.neuron.apply_correction(self.learning_level, rand_y, activation, rand_X)
            
            #calculate training error
            error = calculate_abs_error(self.neuron, X, y)
            
            if error < self.min_error:
                self.min_error = error
                self.min_weights = self.neuron.weights.copy() 
                self.min_bias = self.neuron.bias
                print('updated min_error', self.min_error)

            i += 1
            n += 1

        return i >= limit

    def best_neuron(self):
        return Neuron(self.min_weights, self.min_bias, self.activation_func)

class Neuron:

    def __init__(self, weights, bias, activation_func):
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    # def get_weights_with_bias(self):
    #     w = [self.bias]
    #     w.extend(self.weights)
    #     return np.asarray(w)

    def apply_correction(self, learning_level, expected_result, result, entry):
        correction = learning_level * (expected_result - result)
        #print(f'Correction: {correction} [{expected_result}/{result}]')
        delta_weights = correction * entry
        #print(f'delta_weights: {delta_weights}')
        delta_bias = correction
        #print(f'Old weights: {self.weights}, bias: {self.bias}')
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
