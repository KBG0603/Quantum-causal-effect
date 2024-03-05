#############################################################
# This python code is built on Toby Cubitt's quantinf package on Matlab (https://www.dr-qubit.org/matlab.html). See the package for detailed comments
# In addition to making python version of some useful quantinf fucntions, 
# I created some of my own functions that I use, and some of the qutip functions that are frequently used
##############################################################


import numpy as np
import scipy as sp
from scipy import linalg as lg
from numpy import linalg as LA
from math import pi 
import qutip as qt
from qutip.metrics import tracedist


### define Paulis
Id = np.matrix(np.eye(2))
sx = np.matrix([[0,1],[1,0]])
sy = np.matrix([[0,-1j],[1j,0]])
sz = np.matrix([[1,0],[0,-1]])
###################

#################

def trace_dist(A,B):
   return  tracedist(qt.Qobj(A),qt.Qobj(B))



def Herm(U):
     return np.matrix(np.transpose(np.conj(np.array(U))))


def tensor(*args):

   M = 1;
   for j in range(len(args)):
       if type(args[j]) is tuple:
           for k in range(args[j][1]):
               M = np.kron(M,args[j][0])
       else:
            M = np.kron(M,args[j])
   return M
 #######################


######################## Partial trace ###############
def TrX(p,sys,dim):
    
    sys = np.array(sys)
    dim = np.array(dim)



# check arguments
    if np.any(sys > len(dim)) or np.any(sys <= 0):
      raise Exception('Invalid subsystem in SYS')
   
    if (len(dim) == 1 and np.mod(len(p)/dim,1) != 0) or len(p) != np.prod(dim):
      raise Exception('Size of state PSI inconsistent with DIM');
    
    
    
    # remove singleton dimensions
    sys = np.setdiff1d(sys,np.where(np.array(dim)==1)[0])
   
    dim = dim[np.where(dim != 1)]
    
    
    # calculate systems, dimensions, etc.
    n = len(dim);
    rdim = dim[::-1]
    keep = np.arange(0,len(dim))
    keep = np.setdiff1d(keep,keep[sys-1])
    dimtrace = np.prod(dim[sys-1]);
    dimkeep = int(len(p)/dimtrace);
    
    
    if np.any(np.array(np.shape(p))==1):
      p = np.array(p)
      # state vector
      if np.shape(p)[0] == 1:
        p = np.matrix(p).getH()
      
      # reshape state vector to "reverse" ket on traced subsystems into a bra,
      # then take outer product
      perm = n+1-np.concatenate([keep[::-1]+1,sys])
      x = np.reshape(np.transpose(np.reshape(p,tuple(rdim),'F'),tuple(perm-1)),tuple([int(dimkeep),dimtrace]),'F');
      x = np.matrix(x)
      x = x @ x.getH();
    
    
    else:
      # density matrix
    
      # reshape density matrix into tensor with one row and one column index
      # for each subsystem, permute traced subsystem indices to the end,
      # reshape again so that first two indices are row and column
      # multi-indices for kept subsystems and third index is a flattened index
      # for traced subsystems, then sum third index over "diagonal" entries
      perm = n+1-np.concatenate([keep[::-1]+1,keep[::-1]+1-n,sys,sys-n])
      reshape_I = np.reshape(np.array(p),tuple(np.concatenate([rdim,rdim])), order="F")
      Perm_I = np.transpose(reshape_I,tuple(perm-1))
      reshape_II = np.reshape(Perm_I,tuple([int(dimkeep),int(dimkeep),dimtrace**2]),'F')
      x = np.sum(reshape_II[:,:,np.arange(1,dimtrace**2+1,dimtrace+1)-1],axis=2)
    
    return x

#############################


def applyChan(N, rho):
    #### Takes N as a qutip generated Channel not in Kraus form ### 
    N_1 = qt.to_kraus(N)
    S = 0 
    for ii in range(len(N_1)):
        S = S + np.matrix(N_1[ii]) @ rho @ np.matrix(N_1[ii]).getH()
    return S
####################################



def applyChan_kraus(N, rho):
    ## works when the qutip channel N is already in the Kraus form ####
    S = 0 
    for ii in range(len(N)):
        S = S + np.matrix(N[ii]) @ rho @ np.matrix(N[ii]).getH()
    return S




########################
def view_mat(A):
    import pandas as pd
    A_df = pd.DataFrame(A)
    return A_df
#########################    

######################### Partial transpose ############
def TX(p,sys,dim):
    dim = np.array(dim)
    sys=np.array(sys)
    p=np.array(p)
    n = len(dim)
    d = np.shape(p)
    perm = np.arange(1,2*n+1)
    perm[list(np.concatenate([n-np.array(sys),2*n-np.array(sys)]))] = perm[list(np.concatenate([2*n-np.array(sys),n-np.array(sys)]))]
    x = np.reshape(np.transpose(np.reshape(p,tuple(np.concatenate([dim[::-1],dim[::-1]])),'F'),tuple(perm-1)),d,'F')
    return x
#########################

####
def normalise(p):
    p = np.array(p)
    if np.any(np.array(np.shape(p))==1):
        return np.matrix(p/np.linalg.norm(p))
    else:
        return np.matrix(p/np.trace(p))
#####

############
def ket(basis, dimension):
    I =np.matrix( np.eye(dimension))
    return I[:,basis]
############



def ket2density(psi):
    psi = np.matrix(psi)
    return psi @ psi.getH()

def bell_state(x):
    if x==0:
        return normalise(np.array([[1],[0],[0],[1]]))
    elif x ==1:
        return normalise(tensor((ket(0,2),2))-tensor((ket(1,2),2)))
    elif x==2:
        return normalise( tensor(ket(0,2),ket(1,2)) + tensor(ket(1,2),ket(0,2)) )
    elif x==3:
        return normalise( tensor(ket(0,2),ket(1,2)) - tensor(ket(1,2),ket(0,2)) )
    else:
        print("put an integer value belonging to {0,1,2,3}")
        
def concurrence(p):
    p = np.matrix(p)
    if np.any(np.array(np.shape(p))==1):
        return  2*np.abs(p[0]*p[3] - p[1]*p[2])[0,0]
    else:
        rho_tilde = tensor((sy,2)) @ np.conj(p) @ tensor((sy,2))
        R = lg.sqrtm(lg.sqrtm(p) @ rho_tilde @ lg.sqrtm(p))
        a,b = LA.eig(R)
        return max(0,abs(a[0]-a[1]-a[2]-a[3]))



    
def randPsi(dimension):
    sum = 0
    for i in range(dimension):
        sum = sum + ket(i,dimension)*(np.random.randn()+1j*np.random.randn())
    return normalise(sum)


def inner_prod(a,b):
    a = np.matrix(a) ; b = np.matrix(b)
    return chop((a.getH() @ b)[0,0])
    
        
def outer_prod(a,b):
    a = np.matrix(a) ; b = np.matrix(b)
    a @ b.getH()
    
    
def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q
                            
# def randU(dimension):
#     sum = 0
#     Q = gram_schmidt_columns(ket2density(randPsi(dimension)))
#     for i in range(dimension):
#         sum = sum + np.exp(1j* pi* np.random.randn()) * ket2density(Q[:,i])
#     return sum


################## Half waveplate ##########
def HWP(theta):
    return chop((np.cos(2*theta)*sz + np.sin(2*theta)*sx))



#### Quarter waveplate #####

def QWP(theta):
    y =  np.exp(-1j*np.pi/4)*np.matrix([[np.cos(theta)**2 +1j*np.sin(theta)**2 ,(1-1j)*np.sin(theta)*np.cos(theta)],[(1-1j)*np.sin(theta)*np.cos(theta),np.sin(theta)**2 +1j*np.cos(theta)**2 ]])
    #y=np.sqrt(2)*(Id - 1j*HWP(theta))
    return chop(y)
#####################


def randRho(dimension):
    sum = 0
    Q = gram_schmidt_columns(ket2density(randPsi(dimension)))
    for i in range(dimension):
        sum = sum + np.random.rand()*ket2density(Q[:,i])
    return normalise(sum)


### Random isometery #####
def randV(d_out,d_in):
    X = (np.random.randn(d_out, d_in) + 1j*np.random.randn(d_out, d_in))/np.sqrt(2)
    Q,R = LA.qr(X, mode='reduced')
    R = np.diag(np.diag(R)/np.abs(np.diag(R)))
    return np.matrix(Q@R)


def randU(dimension):
    return randV(dimension,dimension)

def randH(dimension):
    H = np.matrix(2*(np.random.randn(dimension) + 1j*np.random.randn(dimension))-(1+1j))
    return H + H.getH()

    
############ kill small values ######
def chop(expr, delta=10**-10):
    Re =  np.ma.masked_inside(np.real(expr), -delta, delta).filled(0)
    Im =  np.ma.masked_inside(np.imag(expr), -delta, delta).filled(0)
    return Re + Im*1j
##########################

def isQstate(x):
    
    if np.any(np.array(np.shape(x))==1):
        return isQstate(ket2density(x))
    else:
        a,b = LA.eig(x)
        if np.any(a.imag>=10**-7):
            return False
        elif np.any(a.real)<0:
            return False
        elif abs(sum(abs(a))-1)>=10**-5:
            return False
        else:
            return True
    
def isKet(x):
    if isQstate(x) and np.any(np.array(np.shape(x))==1):
        return True
    else:
        return False





    
def syspermute(p,perm,dim):
    
#  SYSPERMUTE  permute order of subsystems in a multipartite state


    perm = np.array(perm)
    dim = np.array(dim)
    n = len(dim);
    d = np.shape(p)    
    if len(perm) != n:
        raise Exception('Number of subsystems in PERM and DIM inconsistent')
    if np.any(np.sort(perm) != np.arange(1,n+1)):
        raise Exception ('PERM is not a valid permutation')
    if len(p) != np.prod(dim):
        raise Exception ('Total dimension in DIM does not match state P')

    if np.any(np.array(np.shape(p))==1):
        # state vector
        perm = n+1-perm[::-1]
        reshape_I = np.reshape(np.array(p),tuple(dim[::-1]),'F')
        permute_I = np.transpose(reshape_I,tuple(perm-1))
        q = np.reshape(permute_I,tuple(d),'F')

    elif d[0] == d[1]:
  # density matrix
        perm = n+1-perm[::-1]
        perm = np.concatenate([perm,n+perm])
        reshape_I = np.reshape(np.array(p),np.tuple(np.concatenate([dim[::-1]],dim[::-1])),'F')
        permute_I = np.transpose(reshape_I,tuple(perm-1))
        q = np.reshape(permute_I,tuple(d),'F')
    return q 

def sysExchange(p,sys,dim):
    
#     SYSEXCHANGE  exchange order of two subsystems in a multipartite state
   
    sys = np.array(sys)
    dim = np.array(dim)
    perm = np.arange(1,len(dim)+1)
    perm[sys[0]-1] = sys[1]
    perm[sys[1]-1] = sys[0]

    return syspermute(p,perm,dim)


def Fidelity(a,b):
    
    if isKet(a) and isKet(b):
        return np.abs(inner_prod(a, b))**2
    elif isKet(a) is False and isKet(b):
        return np.abs(np.matrix(b).getH() @ np.matrix(a) @ np.matrix(b))[0,0]
    elif isKet(b) is False and isKet(a):
        return np.abs(np.matrix(a).getH() @ np.matrix(b) @ np.matrix(a))[0,0]
    else:
        a = np.matrix(a); b = np.matrix(b)
        return abs(np.trace(lg.sqrtm( lg.sqrtm(a) @ b @ lg.sqrtm(a))))**2
    
    
def SWAP(dim):
    s=0
    for i in range(dim):
        for j in range(dim):
            s = s+ tensor(ket(i,dim),ket(j,dim))@tensor(ket(j,dim),ket(i,dim)).getH()
    return s
