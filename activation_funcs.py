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
    beta = 1	
    return np.tanh(beta*excitement)

def lineal_activation(excitement):
    return excitement


def dx_sign_activation(excitement):
    return 0

def dx_tanh_activation(excitement):
    beta = 1
    return beta/np.cosh(beta*excitement)**2
    # return 1 - (tanh_activation(excitement)**2) 

def dx_lineal_activation(excitement):
    return 1