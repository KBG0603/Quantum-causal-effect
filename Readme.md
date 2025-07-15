# Quantum Causal Effect

This repository contains the implementation of variational quantum algorithms for studying quantum causal effects, as described in the paper published at PRX Quantum 6, 020335 (2025). The code provides tools to optimize quantum circuits for estimating quantum causal effects using the PennyLane quantum computing framework.

## Overview

The repository includes two main Python scripts that implement variational quantum algorithms for different quantum causal effect scenarios, along with a utility module `Quantinf_functions.py` for quantum information processing functions:

1. **Github_ACE_Partial_SWAP_varying_input_state_higher_d.py**: Implements a variational algorithm to estimate the maximum quantum causal effect for the superposition of common cause and direct cause, realised by a partial SWAP gate, optimizing over input states and POVM measurements.
2. **ACE_coherent_superposition_channels_github.py**: Implements a variational algorithm to estimate the maximum quantum causal effect for a coherent superposition of quantum channels, specifically depolarizing channels.
3. **Quantinf_functions.py**: A Python module adapted from Toby Cubitt's quantinf Matlab package, with additional custom functions and some Qutip-inspired utilities for quantum information processing.

## Prerequisites

To run the scripts, you need to install the following dependencies:

- Python 3.8+
- PennyLane (`pennylane`)
- NumPy (`numpy`)
- SciPy (`scipy`)
- Pandas (`pandas`)
- Matplotlib (`matplotlib`)
- Seaborn (`seaborn`)

You can install the required packages using pip:

```bash
pip install pennylane numpy scipy pandas matplotlib seaborn
```

The `Quantinf_functions.py` module, included in this repository, provides essential quantum information processing functions. Ensure this file is in the same directory as the main scripts or update the `sys.path.append('xxxx')` line in both scripts to point to its location.

## File Descriptions

### 1. Github_ACE_Partial_SWAP_varying_input_state_higher_d.py

This script implements a variational quantum algorithm to estimate the quantum causal effect for a partial SWAP gate. The algorithm optimizes over input states and POVM measurements to maximize the quantum causal effect.

- **Key Parameters**:

  - `n_qubits`: Number of qubits determining the dimension of the quantum system.
  - `theta`: Angle for the partial SWAP gate.
  - `p`: Purity of the reduced noisy state.
  - `steps`: Number of optimization steps (default: 2000).
  - `eta`: Learning rate for gradient descent (default: 0.5).
  - `Threshold`: Convergence threshold for optimization (default: 10^-5).

- **Functionality**:

  - Defines a partial SWAP gate and a reduced state as a convex mixture of a random pure state and a maximally mixed state.
  - Uses two quantum circuits (`circuit1` and `circuit2`) to compute the difference in probabilities for different input states (|0&gt; and |1&gt;).
  - Optimizes the cost function using gradient descent to maximize the quantum causal effect.
  - Plots the optimization history compared to the theoretical true value.

### 2. ACE_coherent_superposition_channels_github.py

This script implements a variational quantum algorithm to estimate the quantum causal effect for a coherent superposition of depolarizing channels.

- **Key Parameters**:

  - `n_qubits`: Number of qubits determining the dimension of the target quantum system.
  - `pe`: Strength of the depolarizing channel (0 for maximum depolarization, 1 for identity channel).
  - `steps`: Number of optimization steps (default: 2000).
  - `eta`: Learning rate for gradient descent (default: 0.5).
  - `Threshold`: Convergence threshold for optimization (default: 10^-5).
  - `layer_POVM`: Number of layers for POVM optimization (default: 5).

- **Functionality**:

  - Defines Kraus operators for a coherent superposition of depolarizing channels.
  - Uses two quantum circuits (`circuit1` and `circuit2`) to compute the difference in probabilities for different input states.
  - Optimizes the cost function using gradient descent to maximize the quantum causal effect.
  - Plots the optimization history compared to the theoretical value.

### 3. Quantinf_functions.py

This module contains utility functions for quantum information processing, adapted from Toby Cubitt's quantinf Matlab package. It includes Python implementations of key functions, additional custom functions, and some utilities inspired by Qutip. The module is essential for running the main scripts, providing functions like `ket`, `ket2density`, `SWAP`, `randPsi`, and others.

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/KBG0603/Quantum-causal-effect.git
   cd Quantum-causal-effect
   ```

2. **Set Up the Environment**: Ensure all dependencies are installed. The `Quantinf_functions.py` module is included in the repository, so ensure it is in the same directory as the main scripts or update the `sys.path.append('xxxx')` line to point to its location.

3. **Run the Scripts**: Execute the scripts using Python:

   ```bash
   python Github_ACE_Partial_SWAP_varying_input_state_higher_d.py
   python ACE_coherent_superposition_channels_github.py
   ```

4. **Output**:

   - Each script prints the optimization progress, showing the cost function value at each step.
   - A plot is generated at the end, showing the optimization history (`TD_history`) compared to the theoretical value.

## Notes

- The `Quantinf_functions.py` module is included in the repository and must be accessible to the main scripts. Refer to Toby Cubitt's quantinf Matlab package for detailed documentation on the original functions.
- The scripts are configured for a single qubit (`n_qubits = 1`) by default. You can modify `n_qubits` to explore higher-dimensional systems.
- Adjust hyperparameters like `steps`, `eta`, `layer_POVM`, and `layer_st` to improve convergence or explore different scenarios.

## Citation

If you use this code in your research, please cite the following paper:

> Giulio Chiribella, Kaumudibikash Goswami, "Maximum and minimum causal effects of physical processes," PRX Quantum 6, 020335 (2025). DOI: 10.1103/PRXQuantum.6.020335

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or issues, please open an issue on this repository.