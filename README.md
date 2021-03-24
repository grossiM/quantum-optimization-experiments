# quantum-quantum-wrapper
A Python wrapper for quantum algorithms for testing purposes. 

**NOT for production use, merely to simplify testing/practicing with local data**

The wrapper will enable developers to use and test multiple algorithms from a single file which has connectors for each quantum algorithm. 


The requirements for this project are the following: 
- [ ] Which algorithm would you need to include?
- [ ] What are the required inputs and outputs of function?
- [ ] Define any pre-processing or post-processing (ex: prepare quantum states, set initialized values based on input. 
- [ ] Include logging to monitor iterations, messages, job monitoring, etc. (also include those available from within Qiskit) 
- [ ] Clean error/exits when issues arise
- [ ] Clean error/exits when issues arise

### Testing
For unit tests, PyTest will be used. You can read and install PyTest at: https://pypi.org/project/pytest/ 

### Versions 
The versions of Python, Qiskit, and PyTest will be printed out in the last cell of each notebook so to determine the last version used/tested of the current code base. 

There are two forms of printing ouot the versions. 
A full version list, which includes the System information is as follows: 
```
import qiskit.tools.jupyter
%qiskit_version_table
```

To print only the Qiskit version, without system information, you can use the following: 
```
qiskit.__qiskit_version__
```


# Contributors (in alphabetical order)
Michele Grossi, Robert Loredo, Voica Radescu

