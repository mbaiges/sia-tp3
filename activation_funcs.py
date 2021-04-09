import math
import numpy as np

def sign_activation(excitement):
    if excitement > 0:
        return 1
    elif excitement < 0:
        return -1
    else:
        return 0
        
def tanh_activation(excitement):	
    return np.tanh(excitement)

def lineal_activation(excitement):
    return excitement

