import numpy as np

#Neuron

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
        excitation = np.dot(entry, self.weights)
        #print(f"\tExcitation: {excitation}")
        activation = self.activation_func(excitation + self.bias)
        #print(f"\tActivation: {activation}")
        return activation

 