import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from math import floor, ceil, log
from datetime import date
from warnings import warn

from qiskit.finance.data_providers import YahooDataProvider
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
    
# convert datetime as date
to_YM = lambda x: date(x.year, x.month, 1)

#######################################################################################

## relevant functions. Used in the OptScheme

#######################################################################################


def min_div(a, m):
    """
    Find q = argmin_{q in N} |q*a - m|. This q is given by either ceil(m/a) or floor(m/a), depending on the rest.
    """
    if (a <= 0) or (m < a):
        return None
    tmp_d = {
        floor(m/a): m - a * floor(m/a),
        ceil(m/a): a * ceil(m/a) - m
    }
    return min(tmp_d, key=tmp_d.get)

# Resample stocks array
grouping_stocks = lambda s: np.vectorize(min_div)(s, max(s))

def generate_values(stock_tickers, start_date, end_date):
    """
    Given a list of stocks, generates the Yahoo Finance prices DataFrame prices_df. Every price series is a stored in a column. The prices df is resampled on month for the time being, and has the latest monthly datapoint as value.
    :param stock_tickers: list, list of stock names. 
    :start_date: datetime.date or datetime.datetime 
    :end_date: datetime.date or datetime.datetime
    :return: prices, returns mean, returns covariance
    """
    data = YahooDataProvider(tickers=stock_tickers,
                 start = start_date,
                 end = end_date)
    data.run()

    prices = np.array([s.iloc[-1] for s in data._data])
    return prices, data.get_period_return_mean_vector(), data.get_period_return_covariance_matrix()

def qcmodel(prices, k, budget, mu, sigma, q):
    """
    Quantum model. It returns the best stocks allocation and the results dictionary. 
    The whole model is computed inside the function.
    :param prices: np.array, not resampled prices 
    :param k: float, scaling factor 
    :param budget: float, budget
    :param mu: array, expected returns 
    :param sigma: sigma, where sigma is the returns' covariance 
    :param q: risk factor q
    :return: best allocation (for non-resampled stocks), resampling array for given stocks
    """
    # compute Q
    Q = sigma * q
    
    # resample the prices
    grouping = grouping_stocks(prices)
    s = grouping * prices
    budget_bits = floor(np.log2(budget/np.min(s)))
    
    # build the model
    mdl = Model('portfolio_optimization')
    x = mdl.integer_var_list((f'x{i}' for i in range(len(s))), lb=0, ub=2**budget_bits-1)

    objective = mdl.sum([mu[i]*x[i] for i in range(len(s))])
    objective -= mdl.sum([Q[i,j]*x[i]*x[j] for i in range(len(s)) for j in range(len(s))])
    mdl.maximize(objective)

    norm = s.mean()/k
    mdl.add_constraint(mdl.sum(x[i] * ceil(s[i]/norm) for i in range(len(s))) <= floor(budget/norm))
    
    return mdl, grouping 

def print_etfs(etfs, savepath):
    fig = plt.figure(figsize = (15, 10))
    for labelname, etf in etfs.items():
        portf_value = {to_YM(pd.to_datetime(k)): v['liquidity'] + v['portfolio_value'] for k, v in etf.items()}
        fig = plt.plot(portf_value.keys(), portf_value.values(), label = labelname)
    plt.title('Portfolio value comparison')
    plt.xlabel('Time evolution', loc = 'right')
    y = plt.ylabel('Value (EUR)', loc = 'top')
    y.set_rotation(0)
    plt.legend()
    fs = os.path.join(savepath, 'comparison.png')
    plt.savefig(fs)
    plt.close()
    return None   


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
    :param resampling: bool, if True resample the prices
    :return: total number of qubits
    """
    if resampling:
        groups = grouping_stocks(prices)
        grouped_prices = groups * prices
    else:
        grouped_prices = prices 
    
    default_qbits = len(grouped_prices) * floor(log(B/np.min(grouped_prices), 2))
    scaling_qbits = ceil(log(floor(B/np.mean(grouped_prices) * k), 2))
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
    :param resampling: bool, if True resamples the prices
    """
    
    if resampling:
        groups = grouping_stocks(prices)
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
         
    
    errors_d = {i: err(grouped_prices, i) for i in range(1, scale_range + 1)}
    
    if max_qbits is not None: 
        qbits_d = {i: model_qbits(prices, i, B) for i in range(1, scale_range + 1)}
        scale_per_qbit = dict_inverse(qbits_d)
        optimal_values = [j for j in errors_d if errors_d[j] <= alpha]
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

#######################################################################################

## relevant functions. Used in the benckmarking

#######################################################################################


def dates_gen(start, end, delta):
    """
    Given a start date, an end date, and a time delta, it returns a generator of dates
    """
    curr = start
    while curr < end:
        yield curr
        curr += delta


def is_market_data_complete(market_data):
    """
    Returns a boolean stating whether the market data is complete (no missing data for any date) or not
    """
    if isinstance(market_data, list):
        masks = [pd.notna(md) for md in market_data]
        return pd.concat(masks, axis='columns').all(axis=None)
    else:
        return pd.notna(market_data).all()

