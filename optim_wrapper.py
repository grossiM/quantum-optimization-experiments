#!/usr/bin/env python
# coding: utf-8

# Import Grover’s algorithm and components classes
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.operators import PauliExpectation, CVaRExpectation
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToQubo
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit import execute, Aer
from qiskit.aqua import aqua_globals

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import json
from docplex.mp.model import Model

aqua_globals.random_seed = 123456
aqua_globals.massive=True

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# setup aqua logging
from qiskit.aqua import set_qiskit_aqua_logging, get_qiskit_aqua_logging
loglevel = logging.DEBUG # choose INFO, DEBUG to see the log
set_qiskit_aqua_logging(loglevel)  

# Import the QuantumInstance module that will allow us to run the algorithm on a simulator and a quantum computer
from qiskit.aqua import QuantumInstance

# Optimization's dictionary is used to wrap all the necessary parameters in one dictionary. 
# The following is the dictionary we will use for Optimization.
"""
optim_dict = {
  "docplex_mod": 'mdl',
  "quantum_instance": Backend,
  "shots": 1024,
  "print":boolean,
  "logfile":boolean,
  "solver":'method',
  "optimizer":'SPSA',
  "maxiter":'100',
  "depth":'1',
  "alpha":0.35,
  "initial_point":list
}
"""

# Define our Optimisation function. This is the function that will be called by the quantum-aggregator class.
def optimize_portfolio(dictionary):
    #dictionary["expression"]
    
    if dictionary.get('logfile'): 
        old_logging_level = get_qiskit_aqua_logging() # in case externally overwritten
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_folder = f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S.%f')}"
        os.mkdir(log_folder)
        set_qiskit_aqua_logging(loglevel, f"{log_folder}/log.txt")

    result ={}
    # case to 
    qp = QuadraticProgram()
    qp.from_docplex(dictionary['docplex_mod'])

    if dictionary.get('print'):
        print('### Original problem:')
        print(qp.export_as_lp_string())
    

    #classical solution
    # solve classically as reference
    if dictionary['solver'] == 'classic':
        t_00 = time.perf_counter()
        opt_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
        t_0 = time.perf_counter() - t_00
        result['computational_time'] = t_0
        result['result'] = opt_result
        #print('Time:',t_0)
        #print(opt_result)
    elif dictionary['solver'] == 'vqe':
        
        # used for visualization and for the ansatz
        # to get the number of binary variables
        conv = QuadraticProgramToQubo()
        qp1 = conv.convert(qp)
        #This is only for visualization
        if dictionary.get('print'):
            print('### quadratic_program_to_qubo:')
            print(qp1.export_as_lp_string())
            print("Penalty:", conv.penalty)
        
        #quantum preparation
        # set classical optimizer
        optimizer = dictionary["optimizer"](maxiter=int(dictionary["maxiter"]))

        # set variational ansatz
        var_form = RealAmplitudes(qp1.get_num_binary_vars(), reps=int(dictionary["depth"]))
        m = var_form.num_parameters

        # set backend
        backend = Aer.get_backend(dictionary["quantum_instance"])
    
        # initialize CVaR_alpha objective
        cvar_exp = CVaRExpectation(float(dictionary["alpha"]), PauliExpectation())
        cvar_exp.compute_variance = lambda x: [0]  # to be fixed in PR #1373

        # use an initial point for vqe parameters, if given
        initial_point = dictionary.get('initial_point')

        # initialize VQE using CVaR
        vqe = VQE(expectation=cvar_exp, optimizer=optimizer, var_form=var_form,
                  quantum_instance=backend, initial_point= initial_point)

        # initialize optimization algorithm based on CVaR-VQE
        opt_alg = MinimumEigenOptimizer(vqe)

        # solve problem
        t_00 = time.perf_counter()
        results = opt_alg.solve(qp)
        t_0 = time.perf_counter() - t_00
        result['computational_time'] = t_0
        result['eval_count'] = vqe._eval_count # also vqe._eval_time exists
        result['result'] = results
        result['solver_info'] = {'optimal_params' : list(vqe.optimal_params)} # list is json serializable

    result['is_qp_feasible'] = qp.is_feasible(result['result'].x)

    # print results
    if dictionary.get('print'):
        print('### Results:')
        print(result)
        
    if dictionary.get('logfile'):
        with open(f'{log_folder}/dictionary.json', 'w') as fp:
            d = dictionary.copy()
            d['docplex_mod'] = dictionary['docplex_mod'].export_as_lp_string()
            d['optimizer'] = dictionary['optimizer'].__name__
            json.dump(d, fp)
        with open(f'{log_folder}/result.json', 'w') as fp:
            d = result.copy()
            d['result'] = {}
            d['result']['optimal_function_value'] = result['result'].fval
            d['result']['optimal_value'] = list(result['result'].x)
            d['result']['status'] = str(result['result'].status)
            json.dump(d, fp)
        set_qiskit_aqua_logging(old_logging_level) # restores previous logging state
        
    return result