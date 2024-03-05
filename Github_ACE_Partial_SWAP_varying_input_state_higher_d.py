

import pennylane as qml
from pennylane import numpy as np1
import sys
import pandas as pd
#sys.path.append('xxxx') #Type where your relevant files are
from Quantinf_functions import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path  

# program

n_qubits =1 # no.of qubits determines dimension of the input and output qunatum system of the reduced channel
d = 2**n_qubits

theta = np.pi/4 #Theta of the partial swap gate


U_P_Swap = np1.array(np1.cos(theta)*np1.eye(d**2)-1j*np1.sin(theta)*SWAP(d)) #Define partial swap gate: U = exp(i*theta* swap)

dim = [d,d] #denotes dxd input state to the swap gate

p=1 #determines purity of the reduced noisy state at the inaccessible quantum system C 

U_reduced_st = np1.array(p*ket2density(randPsi(d))+(1-p)*np.eye(d)/d) # reduced state, a convex mixture of maximally mixed state and a random pure state


n_F_wire = 1 #Fiduciary wire for POVM optmization
n_wire = 2*n_qubits + n_F_wire #Total input wire: 2 wires for two quantum systems of the partial swap gate and one fiduciary system for POVM optimization




dev = qml.device("default.mixed", wires=n_wire)

@qml.qnode(dev,diff_method="backprop")
def circuit1(th):
    th_resh_st=np1.array(th[0,0:np.prod(shape_st)])
    th_st = th_resh_st.reshape(shape_st) #Filtering out the weight for the input state optimization
    th_resh_POVM=np1.array(th[0,np.prod(shape_st):])
    th_POVM = th_resh_POVM.reshape(shape_POVM) #Filtering out the weight for the POVM optimization
    qml.QubitDensityMatrix(np1.array(ket2density(ket(0,2**n_qubits))), wires=range(n_wire)[0:n_qubits]) #Preparing the input states to be |0>, this input states will be optimized
    qml.QubitDensityMatrix(np1.array(ket2density(ket(0,2**n_F_wire))), wires=range(n_wire)[2*n_qubits:]) # Preparing the fiduciary states in |0>
    qml.QubitDensityMatrix(U_reduced_st, wires=range(n_wire)[n_qubits:2*n_qubits]) #Preparing the reduced state at the inaccessible system due to Alice tracing out her input quantum systems
    qml.StronglyEntanglingLayers(weights=np1.array(th_st), wires=range(n_wire)[0:n_qubits]) #Strongly entangling unitary for input state optimization
    qml.QubitUnitary(U_P_Swap, wires=range(n_wire)[0:2*n_qubits]) #Apply parital swap
    qml.StronglyEntanglingLayers(weights=np1.array(th_POVM), wires=range(n_wire)[n_qubits:]) #Strongly entangling unitary for POVM optimization
    return qml.probs(wires=range(n_wire)[-1]) #measures the last qubit system in the computational basis at the output of the prover




dev2 = qml.device("default.mixed", wires=n_wire)

@qml.qnode(dev2,diff_method="backprop")
def circuit2(th):
    th_resh_st=np1.array(th[0,0:np.prod(shape_st)])
    th_st = th_resh_st.reshape(shape_st) #Filtering out the weight for the input state optimization
    th_resh_POVM=np1.array(th[0,np.prod(shape_st):])
    th_POVM = th_resh_POVM.reshape(shape_POVM) #Filtering out the weight for the POVM optimization
    qml.QubitDensityMatrix(np1.array(ket2density(ket(1,2**n_qubits))), wires=range(n_wire)[0:n_qubits]) #Preparing the input states to be |1>, this input states will be optimized
    qml.QubitDensityMatrix(np1.array(ket2density(ket(0,2**n_F_wire))), wires=range(n_wire)[2*n_qubits:]) # Preparing the fiduciary states in |0>
    qml.QubitDensityMatrix(U_reduced_st, wires=range(n_wire)[n_qubits:2*n_qubits]) #Preparing the reduced state at the inaccessible system due to Alice tracing out her input quantum systems
    qml.StronglyEntanglingLayers(weights=np1.array(th_st), wires=range(n_wire)[0:n_qubits]) #Strongly entangling unitary for input state optimization
    qml.QubitUnitary(U_P_Swap, wires=range(n_wire)[0:2*n_qubits]) #Apply parital swap
    qml.StronglyEntanglingLayers(weights=np1.array(th_POVM), wires=range(n_wire)[n_qubits:]) #Strongly entangling unitary for POVM optimization
    return qml.probs(wires=range(n_wire)[-1]) #measures the last qubit system in the computational basis at the output of the prover



def cost_fn(x):
    return circuit1(x)[0]-circuit2(x)[0] 


# main hyperparameters for variational optimisation 

Threshold=10**(-5)
steps = 2000  # Number of optimization steps
eta = 0.5  # Learning rate
q_delta = 1  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator
#np.random.seed(rng_seed)



layer_POVM = 2
wires_POVM = n_qubits+n_F_wire
shape_POVM = qml.StronglyEntanglingLayers.shape(n_layers=layer_POVM, n_wires=wires_POVM)
w_povm = q_delta * np1.random.random(size=shape_POVM)

w_resh_POVM = w_povm.reshape(1,np.prod(shape_POVM))

layer_st=2
wires_state = n_qubits
shape_st = qml.StronglyEntanglingLayers.shape(n_layers=layer_st, n_wires=wires_state)
w_st = q_delta * np1.random.random(size=shape_st)

w_resh_st = w_st.reshape(1,np.prod(shape_st))

w=np1.concatenate((w_resh_st,w_resh_POVM),axis=1) #Prepare the concatenated weight for both input state and POVM optimization

opt = qml.GradientDescentOptimizer(eta)

cost_history = []
for it in range(steps):
    w, cost = opt.step_and_cost(cost_fn, w)
    print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    if np.abs(np.mean(cost_history[-10:])-cost)<=Threshold:
        break
    cost_history.append(cost)

    

TD_history = [-cost_history[iter] for iter in range(len(cost_history))]



true_value = np.sin(theta)*np.sqrt(np.sin(theta)**2+p**2*np.cos(theta)**2) #True value according to our Lemma
  


import seaborn as sns
plt.style.use("seaborn")
sns.set(font_scale=1.5)
plt.plot(TD_history, "bo-")
plt.plot(true_value*np.ones(len(cost_history)),'r-.')
plt.ylabel("qCE$_{\max}$", fontsize=20)
plt.xlabel("Optimization steps", fontsize=20)
plt.show()