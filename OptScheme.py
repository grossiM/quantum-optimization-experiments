from utils import * 
import argparse 

def main(options): 
    
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

        if options.qbits_limit == 'true':
            qbits_dict = {k0: max(l0) for k0, l0 in dict_inverse({j: model_qbits(prices, j, B) for j in range(1, 100)}).items()}
            if options.max_qbits in qbits_dict: 
                k_ = qbits_dict[options.max_qbits]
            else: 
                warn(f'Number of qbits given not found among possible models.\nThe minimum number of qbits is f{min(qbits_dict.keys())}.\nChoosing default k')
                k_ = options.k
        else:
            k_ = options.k


        mdl, grouping = qcmodel(prices, k_, B, mu, sigma, options.q)
        optim_dict["docplex_mod"] = mdl
        results = aggregator('optimizer', optim_dict)

        # Integer results (amount of groups of stocks): x
        x_val = [results['result'].variables_dict[f'x{i}'] for i in range(len(prices))]

        # Amount of individual stock
        best_allocation = grouping*np.array(x_val)
        budget_spent = np.sum(best_allocation * prices)

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
           
        fs = os.path.join(options.savepath, 'quantum_etf_results.json')
        with open(fs, 'wt') as qf:
            json.dump(quantum_etf, qf)

        # next datapoint
        date_ = next_month(date_)
            
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
    parser.add_argument('-ql', '--qbits_limit', type = str, default = 'true', 
                        help = 'Qubits constraint. If True, attempts to keep the size of the model to fixed qubits')
    parser.add_argument('-s', '--savepath', type = str, default = '', 
                        help = 'Folder where to save the etf. Default none.')
    
    options = parser.parse_args()
    
    main(options)