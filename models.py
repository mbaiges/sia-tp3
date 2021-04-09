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

	def apply_correction(self, learning_level, expected_result, activation, input):
		delta_weight = 

    def evaluate(self, entry):
        print('Evaluating:', entry)
        excitation = self.entry * np.transpose(self.weights)
		print(f"\tExcitation: {excitation}")
		activation = self.activation_func(excitation - self.bias)
        print(f"\tActivation: {activation}")
		return activation
		
 