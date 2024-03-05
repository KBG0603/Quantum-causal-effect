

import pennylane as qml
from pennylane import numpy as np1
import sys

#sys.path.append('xxxx') # Type your path where you put the quantinf fucntions


from Quantinf_functions import *
import matplotlib.pyplot as plt


from pathlib import Path  
import pandas as pd


# program

n_qubits = 1 # Total number of qubits deciding the dimension of the target quantum system, change this to go to higher dimensional target 
d = 2**n_qubits # Dimension of the target system

kraus_op= tensor(([Id,sx,sy,sz],n_qubits)) #Pauli operator basis of n qubit systems

pe=0 # Strength of deplarizing channels, 0 means maximum strength, 1 means identity channel
kr_coeff_depol = [np.sqrt(pe+(1-pe)/(d**2)), np.sqrt(1-pe)/d]   # Kraus operator for depolarizing channel

kraus_op[0] = kraus_op[0]*kr_coeff_depol[0] # Kraus operator with control system
kraus_op[1:]= kraus_op[1:]*kr_coeff_depol[1] # Kraus operator with control system


kraus_op_2 = []
for i in range(len(kraus_op)):
    for j in range(len(kraus_op)):
        kraus_op_2.append(tensor(ket2density(ket(0,2)),np.matrix(kraus_op[i]))+tensor(ket2density(ket(1,2)),np.matrix(kraus_op[j])))


k_list = [1/(2**n_qubits)*np.array(kraus_op_2)[i] for i in range(len(kraus_op_2))] # Normalised Kraus operator for coherent superposition of channels


n_F_wire = 1 # One fiduciary qubit for POVM optimization
n_wire = n_qubits + n_F_wire +1 #extra one quantum system for the control qubit



dev = qml.device("default.mixed", wires=n_wire)

@qml.qnode(dev,diff_method="backprop")
def circuit1(th):
    th_resh_st=np1.array(th[0,0:np.prod(shape_st)])
    th_st = th_resh_st.reshape(shape_st) #Filter out the weights for the strongly entanling layer for the input target system
    th_resh_POVM=np1.array(th[0,np.prod(shape_st):])
    th_POVM = th_resh_POVM.reshape(shape_POVM)     #Filter out the weights for the strongly entanling layer for the POVM optimization
    qml.BasisState(np.zeros(n_wire), wires=range(n_wire)) # First preapare all |0> states
    qml.Hadamard(wires=[0]) # Change control qubit to |+> state using a Hadamard gate
    qml.StronglyEntanglingLayers(weights=np1.array(th_st), wires=range(n_wire)[1:n_qubits+1]) # Strongly entangling layer on all target quantum systems, to scan over all input states
    qml.QubitChannel(k_list, wires=range(n_wire)[0:n_qubits+1]) #Apply the coherent superposition of channels on both target and control
    qml.StronglyEntanglingLayers(weights=np1.array(th_POVM), wires=range(n_wire)) # Another strongly entangling layer to optimize over the POVMs
    return qml.probs(wires=range(n_wire)[0]) #measures the first qubit system in the computational basis at the output of the prover




dev2 = qml.device("default.mixed", wires=n_wire)


@qml.qnode(dev2,diff_method="backprop")
def circuit2(th):
    th_resh_st=np1.array(th[0,0:np.prod(shape_st)])
    th_st = th_resh_st.reshape(shape_st) #Filter out the weights for the strongly entanling layer for the input target system
    th_resh_POVM=np1.array(th[0,np.prod(shape_st):])
    th_POVM = th_resh_POVM.reshape(shape_POVM)     #Filter out the weights for the strongly entanling layer for the POVM optimization
    qml.BasisState(np.zeros(n_wire), wires=range(n_wire)) # First preapare all |0> states
    qml.Hadamard(wires=[0]) # Change control qubit to |+> state using a Hadamard gate
    qml.broadcast(unitary=qml.PauliX, pattern="single", wires=range(n_wire)[1:n_qubits+1]) #Flip all the target systems to make it |1>
    qml.StronglyEntanglingLayers(weights=np1.array(th_st), wires=range(n_wire)[1:n_qubits+1]) # Strongly entangling layer on all target quantum systems, to scan over all input states
    qml.QubitChannel(k_list, wires=range(n_wire)[0:n_qubits+1]) #Apply the coherent superposition of channels on both target and control
    qml.StronglyEntanglingLayers(weights=np1.array(th_POVM), wires=range(n_wire)) # Another strongly entangling layer to optimize over the POVMs
    return qml.probs(wires=range(n_wire)[0]) #measures the first qubit system in the computational basis at the output of the prover





def cost_fn(x):
    return circuit1(x)[0]-circuit2(x)[0] 


# main hyperparameters for variational optimisation 

Threshold=10**(-5)
steps = 2000  # Number of optimization steps
eta = 0.5  # Learning rate
q_delta = 1  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator
#np.random.seed(rng_seed)

layer_POVM = 5 # Play with these layers
wires_POVM = n_wire
shape_POVM = qml.StronglyEntanglingLayers.shape(n_layers=layer_POVM, n_wires=wires_POVM)
w_povm = q_delta * np1.random.random(size=shape_POVM)

w_resh_POVM = w_povm.reshape(1,np.prod(shape_POVM))

layer_st=2 # Play with these layers
wires_state = n_qubits
shape_st = qml.StronglyEntanglingLayers.shape(n_layers=layer_st, n_wires=wires_state)
w_st = q_delta * np1.random.random(size=shape_st)

w_resh_st = w_st.reshape(1,np.prod(shape_st))

w=np1.concatenate((w_resh_st,w_resh_POVM),axis=1) # Create the overall concatenated weights


opt = qml.GradientDescentOptimizer(eta)

cost_history = []
for it in range(steps):
    w, cost = opt.step_and_cost(cost_fn, w)
    print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    if np.abs(np.mean(cost_history[-1:])-cost)<=Threshold: #iteration stops if the absolute difference with respect to the last cost is less than threshold (usually 10^-5)
        break
    cost_history.append(cost)

    

TD_history = [-cost_history[iter] for iter in range(len(cost_history))]

## Plot ##
import seaborn as sns
plt.style.use("seaborn")
sns.set(font_scale=1.5)
plt.plot(TD_history, "bo-")
plt.plot(pe*np.ones(len(cost_history)),'r-.')
plt.ylabel("qCE$_{\max}$", fontsize=20)
plt.xlabel("Optimization steps", fontsize=20)
plt.show()






