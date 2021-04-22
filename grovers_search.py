#!/usr/bin/env python
# coding: utf-8

# Import Grover’s algorithm and components classes
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit import BasicAer

# Import the QuantumInstance module that will allow us to run the algorithm on a simulator and a quantum computer
from qiskit.aqua import QuantumInstance

# Grover's dictionary is used to wrap all the necessary parameters in one dictionary. 
# The following is the dictionary we will use for Grover's Search.
"""
grovers_dict = {
  "expression": 'boolean_expression',
  "quantum_instance": Backend,
  "shots": 1024
}
"""

# Define our Grover's search function. This is the function that will be called by the quantum-aggregator class.
def grovers_search(dictionary):
    oracle = LogicalExpressionOracle(dictionary["expression"])
    quantum_instance = QuantumInstance(dictionary["quantum_instance"], shots=dictionary["shots"])
    grover = Grover(oracle)
    result = grover.run(quantum_instance)
    return result
