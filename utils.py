import json, os, sys, re, math

import pandas as pd
import numpy as np


from math import gcd, floor, ceil, log
from datetime import date
from functools import reduce
from copy import copy
from warnings import warn

from qiskit.finance import QiskitFinanceError
from qiskit.finance.data_providers import YahooDataProvider

from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.aqua.components.optimizers import SLSQP, COBYLA
from IPython.display import clear_output

from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *

from docplex.mp.model import Model


import grovers_search as grovers
import optim_wrapper as optimization

# IBM wrapper

def aggregator(algorithm, dict_details):
    if algorithm == 'grovers':
        result = grovers.grovers_search(dict_details)
    if algorithm == 'optimizer':
        result = optimization.optimize_portfolio(dict_details)
    return result


#######################################################################################

## utility functions

#######################################################################################

def dict_inverse(x):
    """
    Compute the inverse of a dictionary.
    """
    inv_map = {}
    for k, v in x.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    return inv_map

def next_month(date_):
    """
    Given a date, returns the first day of the next month.
    """
    if (date_.month == 12):
        return date(date_.year + 1, 1, 1)
    else:
        return date_.replace(month=date_.month + 1, day=1)

def previous_month(date_):
    """
    Given a date, returns the first day of the previous month.
    """
    if (date_.month == 1):
        return date(date_.year - 1, 12, 1)
    else:
        return date_.replace(month=date_.month - 1, day=1)


#######################################################################################

## other functions. Not directly used in OptScheme

#######################################################################################



def model_qbits(prices, k, B, resampling = True):
    """
    Computes the number of qubits needed for the model.
    The prices are resampled within the function.
    :param prices: np.array, prices 
    :param k: float, scale factor 
    :param B: float, budget 
    :return: total number of qubits
    """
    if not resampling:
        groups = vmin_div(prices, max(prices))
        grouped_prices = groups * prices
    else:
        grouped_prices = prices 
    
    default_qbits = len(grouped_prices) * floor(log(B/np.mean(grouped_prices), 2))
    scaling_qbits = ceil(log(B/np.mean(grouped_prices) * k, 2))
    return default_qbits + scaling_qbits

def generate_scale(prices, B, alpha, scale_range = 100, max_qbits = None, resampling = True):
    """
    Computes the optimal scale at a given approximation threshold alpha for a given price and budget. 
    If max_qbits is given, it returns the best k for the 
    :param prices: np.array, prices 
    :param B: float, budget 
    :param alpha: float, error threshold
    :param scale_range: int, range for the optimal scale. default = 100
    :param max_qbits: int, if not None puts a threshold choosing the optimal scale below the given qbits number
    """
    
    if not resampling:
        groups = vmin_div(prices, max(prices))
        grouped_prices = groups * prices
    else:
        grouped_prices = prices 
    
    # lambda function to compute the scaled prices 
    scale_fact = lambda s, k:  np.vectorize(ceil)(s/np.mean(s) * k)

    # error function, it returns the relative error between prices s and their approximation
    def err(s, k): 
        approx_s = scale_fact(s, k) * np.mean(s) / k
        abs_error = s - approx_s 
        rel_error = abs_error/s
        return max(np.abs(rel_error))
         
    
    errors_d = {i: err(grouped_prices, i) for i in range(1, scale_range+1)}
    
    if max_qbits is not None: 
        qbits_d = {i: model_qbits(prices, i, B) for i in range(1, scale_range+1)}
        scale_per_qbit = dict_inverse(qbits_d)
        optimal_values = [j for j in errors_d if errors_d[j]<=alpha]
        if len(optimal_values) == 0: 
            warn('Attention! No value below given threshold. Returning best k for the given max qubits.')
            if max_qbits in scale_per_qbit:
                return min(scale_per_qbit[max_qbits])
            else:
                warn('Attention! No model with this amount of qubits found. Returning scale = 1')
                return 1
        else: 
            opt_val = min(optimal_values)
            return max(scale_per_qbit[qbits_d[opt_val]])
    else:
        optimal_values = [j for j in errors_d if errors_d[j]<=alpha]
        if len(optimal_values) == 0: 
            warn('Attention! No value below given threshold. Returning k=max')
            return scale_range
        else:
            return min(optimal_values)
        
def random_model(s, b):
    """
    Given prices, budget, mean vector and covariance, it returns a random allocation of them.
    It assumes prices and budget to be already resampled.
    """
    if len(s) == 0:
        return []
    else:
        if b>=s[-1]:
            v = np.random.randint(int(b/s[-1]))
            return random_model(s[:-1], b-v*s[-1]) + [v]
        else:
            v = 0
            return random_model(s[:-1], 1) + [v]
        
        