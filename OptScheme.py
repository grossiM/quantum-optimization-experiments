from utils import date, generate_values, previous_month, dict_inverse, model_qbits, qcmodel, next_month, aggregator, print_etfs

import os
import json
import argparse
import numpy as np

from copy import copy
from math import floor
from warnings import warn
from datetime import datetime
from IPython.display import clear_output

from qiskit.aqua.components.optimizers import SLSQP, COBYLA

def main(options):

    if options.initial_point.lower()=='true' and options.qbits_limit.lower()!='true':
        raise Exception("Can't use initial_point if the number of required qubits varies from one optimization step to the following. Set also qbits_limit to true.")

    # fixed values
    stock_tickers = ['FXD', 'FXR', 'FXL', 'FTXR', 'QTEC']
    start_date = date(2017, 1, 1)
    computation_date = date(2020, 1, 1)
    end_date = date(2021, 1, 1)

    #hyperparameters
    optim_dict = {
          "quantum_instance": 'qasm_simulator',
          "shots": 1024,
          "print": True,
          "logfile": True,
          "solver": 'vqe',
          "optimizer": SLSQP,
          "maxiter": 1000,
          "depth": 1,
          "alpha": 0.35
        }
    # etf collector
    etfs = {}

    # building the quantum etf
    quantum_etf = {}
    date_ = copy(computation_date)

    while date_ < end_date:
        # generate main values for the model
        prices_, mu_, sigma_ = generate_values(stock_tickers = stock_tickers + ['IFV'], start_date = start_date,
                                        end_date = date_)
        # values for quantum etf
        prices = prices_[:-1]; mu = mu_[:-1]; sigma = sigma_[:-1, :-1]


        # find budget, best allocation, results
        if len(quantum_etf) == 0:
            B = copy(options.budget)
        else:
            previous_month_etf = copy(quantum_etf[previous_month(date_).strftime('%Y-%m-%d')])
            B = previous_month_etf['liquidity'] + np.sum(np.array(previous_month_etf['allocation']) * prices)

            if options.initial_point.lower() == 'true':
                print("Using best parameters of previous optimization as an initial point")
                optim_dict['initial_point'] = last_optimal_params

        if options.qbits_limit.lower() == 'true':
            qbits_dict = {k0: max(l0) for k0, l0 in dict_inverse({j: model_qbits(prices, j, B) for j in range(1, options.max_k)}).items()}
            if options.max_qbits in qbits_dict:
                k_ = qbits_dict[options.max_qbits]
            elif options.max_qbits > max(qbits_dict):
                k_ = copy(options.max_k)
                warn("Number of qbits provided exceeds qbits dictionary. Initialising k as the maximum given.")
            else:
                warn('Number of qbits given not found among possible models.Choosing k equal to given value')
                k_ = copy(options.k)
        else:
            k_ = copy(options.k)


        mdl, grouping = qcmodel(prices, k_, B, mu, sigma, options.q)
        optim_dict["docplex_mod"] = mdl

        for i in range(options.n_trials):
            results = aggregator('optimizer', optim_dict)
            if results['is_qp_feasible']:
                break

        # Integer results (amount of groups of stocks): x
        x_val = [results['result'].variables_dict[f'x{i}'] for i in range(len(prices))]

        # Amount of individual stock
        best_allocation = grouping*np.array(x_val)
        budget_spent = np.sum(best_allocation * prices)

        # Saves optimal parameters as initial point for next iteration
        last_optimal_params = results['solver_info']['optimal_params']

        #printing tmp_results
        tmp_results = {
            'computational_time': float(results['computational_time']),
            'optimal_function_value': float(results['result'].fval),
            'status': str(results['result'].status).split('.')[-1],
            'is_qp_feasible': results['is_qp_feasible']
        }

        # generate etf datapoint
        if budget_spent > B: # if budget spent is bigger than current liquidity
            tmp_results['wrong_results'] = {
                'best_allocation_found': [int(i) for i in best_allocation],
                'budget_spent': float(budget_spent)
                }
            if len(quantum_etf) == 0: # at step 0, etf does not spend any budget
                quantum_etf[date_.strftime('%Y-%m-%d')] = {
                    'allocation': [0] * len(prices),
                    'prices': [float(p) for p in prices],
                    'liquidity': B,
                    'portfolio_value': 0.,
                    'results': tmp_results,
                    'k': k_
                }
            else: # at step k (any k), etf keeps previous allocation and updates its values with current prices
                quantum_etf[date_.strftime('%Y-%m-%d')] = {
                    'allocation': previous_month_etf['allocation'],
                    'prices': [float(p) for p in prices],
                    'liquidity': previous_month_etf['liquidity'],
                    'portfolio_value': float(np.sum(np.array(previous_month_etf['allocation']) * prices)),
                    'results': tmp_results,
                    'k': k_
                }
        else: # if budget spent < B
            tmp_results['wrong_results'] = None
            quantum_etf[date_.strftime('%Y-%m-%d')] = {
                'allocation': [int(i) for i in best_allocation],
                'prices': [float(p) for p in prices],
                'liquidity': float(B - budget_spent),
                'portfolio_value': float(budget_spent),
                'results': copy(tmp_results),
                'k': k_
                }

        if not os.path.exists(options.savepath):
            os.makedirs(options.savepath)
        fs = os.path.join(options.savepath, f'quantum_etf_results.json')
        with open(fs, 'wt') as qf:
            json.dump(quantum_etf, qf)

        # next datapoint
        date_ = next_month(date_)

    # building the optimum and real etf
    # steps separated for clarity only
    opt_etf = {}; real_etf = {}
    date_ = copy(computation_date)

    while date_ < end_date:
        # generate main values for the model
        prices_, mu_, sigma_ = generate_values(stock_tickers = stock_tickers + ['IFV'], start_date = start_date,
                                        end_date = date_)
        # values for quantum etf
        prices = prices_[:-1]; mu = mu_[:-1]; sigma = sigma_[:-1, :-1]

        # price of real etf
        ifv_price = prices_[-1]

        # find budget, best allocation, results
        if len(opt_etf) == 0:
            B = copy(options.budget)

            # real values
            no_real_bought_stocks = floor(B/ifv_price)
            real_liquidity = B - no_real_bought_stocks * ifv_price

        else:

            previous_month_etf = copy(opt_etf[previous_month(date_).strftime('%Y-%m-%d')])
            B = previous_month_etf['liquidity'] + np.sum(np.array(previous_month_etf['allocation']) * prices)

        if options.qbits_limit == 'true':
            qbits_dict = {k0: max(l0) for k0, l0 in dict_inverse({j: model_qbits(prices, j, B) for j in range(1, options.max_k)}).items()}
            if options.max_qbits in qbits_dict:
                k_ = qbits_dict[options.max_qbits]
            elif options.max_qbits > max(qbits_dict):
                k_ = copy(options.max_k)
                warn("Number of qbits provided exceeds qbits dictionary. Initialising k as the maximum given.")
            else:
                warn('Number of qbits given not found among possible models.Choosing k equal to given value')
                k_ = copy(options.k)
        else:
            k_ = copy(options.k)

        mdl, grouping = qcmodel(prices, k_, B, mu, sigma, options.q)

        # solver
        mdl.solve()
        sols_dict = dict(zip([f'x{i}' for i in range(len(prices))], [0] * len(prices)))
        for j, v in mdl.solution.as_name_dict().items():
            sols_dict[j] = int(v)

        x_val = list(sols_dict.values())

        # Amount of individual stock
        best_allocation = grouping*np.array(x_val)
        budget_spent = np.sum(best_allocation * prices)

        # generate etf datapoint
        if budget_spent > B: # if budget spent is bigger than current liquidity
            if len(opt_etf) == 0: # at step 0, etf does not spend any budget
                opt_etf[date_.strftime('%Y-%m-%d')] = {
                    'allocation': [0] * len(prices),
                    'prices': [float(p) for p in prices],
                    'liquidity': B,
                    'portfolio_value': 0.,
                    'objective_value': mdl.solution.get_objective_value()
                }
            else: # at step k (any k), etf keeps previous allocation and updates its values with current prices
                opt_etf[date_.strftime('%Y-%m-%d')] = {
                    'allocation': previous_month_etf['allocation'],
                    'prices': [float(p) for p in prices],
                    'liquidity': previous_month_etf['liquidity'],
                    'portfolio_value': float(np.sum(np.array(previous_month_etf['allocation']) * prices)),
                    'objective_value': mdl.solution.get_objective_value()
                }
        else: # if budget spent < B
            opt_etf[date_.strftime('%Y-%m-%d')] = {
                'allocation': [int(i) for i in best_allocation],
                'prices': [float(p) for p in prices],
                'liquidity': float(B - budget_spent),
                'portfolio_value': float(budget_spent),
                'objective_value': mdl.solution.get_objective_value()
                }

        # real etf
        real_etf[date_.strftime('%Y-%m-%d')] = {
            'liquidity': real_liquidity,
            'prices': float(ifv_price),
            'portfolio_value': no_real_bought_stocks * ifv_price
        }


        # etf is saved at every step
        fs_real = os.path.join(options.savepath, f'real_etf_results.json')
        with open(fs_real, 'wt') as rf:
             json.dump(real_etf, rf)

        fs_opt = os.path.join(options.savepath, f'opt_etf_results.json')
        with open(fs_opt, 'wt') as rf:
             json.dump(opt_etf, rf)

        # next datapoint
        date_ = next_month(date_)

        # clearing notebook output
        clear_output()

    etfs['quantum'] = copy(quantum_etf)
    etfs['optimum'] = copy(opt_etf)
    etfs['IFV'] = copy(real_etf)

    print_etfs(etfs, savepath=options.savepath)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Optimization scheme for quantum ETF', usage = 'OptScheme.py')
    parser.add_argument('-q', '--q', type = float, default = 1.,
                        help = 'Risk factor q')
    parser.add_argument('-b', '--budget', type = float, default = 1760.,
                        help = 'Initial budget')
    parser.add_argument('-k', '--k', type = float, default = 7.,
                        help = 'Resampling factor')
    parser.add_argument('-mq', '--max_qbits', type = int, default = 27,
                        help = 'Number of max qubits for the model')
    parser.add_argument('-mk', '--max_k', type = int, default = 100,
                        help = 'Number of max k for the model.')
    parser.add_argument('-ql', '--qbits_limit', type = str, default = 'true',
                        help = 'Qubits constraint. If True, attempts to keep the size of the model to fixed qubits')
    parser.add_argument('-p', '--initial_point', type = str, default = 'false',
                        help = 'If True, it uses the optimal parameters of previous optimization as a starting point for the following')
    parser.add_argument('-t', '--n_trials', type = int, default = 1,
                        help = 'Number of subsequent trials performed in case of INFEASIBLE result')
    parser.add_argument('-s', '--savepath', type = str, default = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_quantum_ETF",
                        help = 'Folder where to save the etf. It is created as default')

    options = parser.parse_args()

    main(options)
