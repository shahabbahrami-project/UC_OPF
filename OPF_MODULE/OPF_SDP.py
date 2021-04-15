# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:36:02 2021
@author: bahramis
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:08:39 2021
@author: bahramis
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:15:33 2021
@author: bahramis
"""
from Ybus import Ybus
from pypower.api import case14                                                                                                                                               as case
from pypower.loadcase import loadcase 
import numpy as np
import cvxpy as cp
from scipy import sparse
def OPF(ppc, PloadProfile, QloadProfile, u):
    Yb= Ybus(ppc)                 #Ybus Matrix 
    nbus=Yb.shape[0]              #number of buses
    ngen=ppc['gen'].shape[0]
    ngenInfo=ppc['gen'].shape[1]
    genMatrix=np.zeros((nbus,ngenInfo))
    for i in range(ngen):
        genMatrix[int(ppc['gen'][i][0])-1,:]=ppc['gen'][i]
    genBusIndex=ppc['gen'][:,0].astype(int)-1
    ck2=ppc['gencost'][:,4]
    ck1=ppc['gencost'][:,5]
    ck0=ppc['gencost'][:,6]   
    baseMVA=ppc['baseMVA']
    Pload=np.zeros((nbus))
    Qload=np.zeros((nbus))
    Pload=PloadProfile/baseMVA
    Qload=QloadProfile/baseMVA
    Qmax=genMatrix[:,3]/baseMVA
    Qmin=genMatrix[:,4]/baseMVA
    Pmax=genMatrix[:,8]/baseMVA
    Pmin=genMatrix[:,9]/baseMVA
    Vmin=0.94*np.ones(nbus)
    Vmax=1.06*np.ones(nbus)
    e=np.eye(nbus, dtype=int)
    Yac=np.zeros((nbus,nbus,nbus), dtype=complex)
    OnesMatrix=np.zeros((nbus,nbus,nbus))
    for k in range(nbus):
        a=e[:,k].reshape(-1,1)  
        b=e[k,:].reshape(1,-1)
        OnesMatrix[:,:,k]=np.matmul(a,b)
        Yac[:,:,k]=np.matmul(OnesMatrix[:,:,k],Yb.todense())
    
    Y=np.zeros((2*nbus,2*nbus,2*nbus))
    Ybar=np.zeros((2*nbus,2*nbus,2*nbus))
    M=np.zeros((2*nbus,2*nbus,2*nbus))
    Z=np.zeros((nbus,nbus,nbus))
    # print(Yb)
    # Y=sparse.csr_matrix(np.zeros((2*nbus,2*nbus,2*nbus)))
    # Ybar=sparse.csr_matrix(np.zeros((2*nbus,2*nbus,2*nbus)))
    # M=sparse.csr_matrix(np.zeros((2*nbus,2*nbus,2*nbus)))
    # Z=sparse.csr_matrix(np.zeros((nbus,nbus,nbus)))
    for k in range(nbus):
        a1=np.real(Yac[:,:,k]+np.transpose(Yac[:,:,k])) 
        a2=np.imag(np.transpose(Yac[:,:,k])-Yac[:,:,k]) 
        a3=np.imag(Yac[:,:,k]-np.transpose(Yac[:,:,k]))
        a4=np.real(Yac[:,:,k]+np.transpose(Yac[:,:,k]))
        Y[:,:,k]=0.5*np.bmat([[a1, a2], [a3, a4]])
    
    for k in range(nbus):
        a1=np.imag(Yac[:,:,k]+np.transpose(Yac[:,:,k]))
        a2=np.real(Yac[:,:,k]-np.transpose(Yac[:,:,k]))
        a3=np.real(np.transpose(Yac[:,:,k])-Yac[:,:,k]) 
        a4=np.imag(Yac[:,:,k]+np.transpose(Yac[:,:,k]))
        Ybar[:,:,k]=-0.5*np.bmat([[a1, a2], [a3, a4]])
    
    for k in range(nbus):
        M[:,:,k]=np.bmat([[OnesMatrix[:,:,k], Z[:,:,k]], [Z[:,:,k], OnesMatrix[:,:,k]]])
    
    
    W = cp.Variable((2*nbus,2*nbus), symmetric=True)
    a = cp.Variable((ngen,1))
    DeltaP=cp.Variable((nbus))
    weight=1000
    constraints = [W >> 0]
    constraints += [
        cp.trace(Y[:,:,k] @ W) <= u[k]*Pmax[k]-(Pload[k]+DeltaP[k]) for k in range(nbus)
    ]
    constraints +=[
        cp.trace(Y[:,:,k] @ W) >= u[k]*Pmin[k]-(Pload[k]+DeltaP[k]) for k in range(nbus)
        ]
    constraints +=[
        cp.trace(Ybar[:,:,k] @ W) <= Qmax[k]-Qload[k] for k in range(nbus)
        ]
    constraints +=[
        cp.trace(Ybar[:,:,k] @ W) >= Qmin[k]-Qload[k] for k in range(nbus)
        ]
    
    constraints +=[
        cp.trace(M[:,:,k] @ W) >= Vmin[k]**2 for k in range(nbus)
        ]
    
    constraints +=[
        cp.trace(M[:,:,k] @ W) <= Vmax[k]**2 for k in range(nbus)
        ]
    constraints +=[
        cp.bmat([[ck1[np.where(genBusIndex ==k)[0][0]]*(cp.trace(Y[:,:,k] @ W)+Pload[k]+DeltaP[k])*baseMVA-a[np.where(genBusIndex ==k)[0][0]]-ck0[np.where(genBusIndex ==k)[0][0]],np.sqrt(ck2[np.where(genBusIndex ==k)[0][0]])*(cp.trace(Y[:,:,k] @ W)+Pload[k]+DeltaP[k])*baseMVA],[np.sqrt(ck2[np.where(genBusIndex ==k)[0][0]])*(cp.trace(Y[:,:,k] @ W)+Pload[k]+DeltaP[k])*baseMVA,-1]])<<0  for k in genBusIndex
        ]
    prob = cp.Problem(cp.Minimize(cp.sum(a)+weight*cp.norm(DeltaP, 1)*baseMVA),
                      constraints)
    
    prob.solve(verbose=True, warm_start=True)

    # Print result.
    print("..............................")
    print("Number of buses is", nbus)
    print("...............................")
    print("Number of generators is", ngen)
    print("...............................")
    print("The optimal value (total generation cost plus penalty) is")
    print(round(prob.value,3), "= ", round(cp.sum(a).value, 3), " $", " + ", round((weight*cp.norm(DeltaP, 1)*baseMVA).value,2), "penalty")
    print("The solution to power level of the generators is")
    for k in genBusIndex:
        print(np.round(((cp.trace(Y[:,:,k] @ W)+Pload[k]+DeltaP[k])*baseMVA).value,3), "MW")
    print("...............................")
    print("The solution to voltage of buses is")
    for k in range(nbus):
        print(np.round(cp.sqrt(cp.trace(M[:,:,k] @ W)).value,3), "pu")
    print("...............................")
    print("The solutions to DeltaP is")
    for k in range(nbus):
        print(np.round((DeltaP[k]*baseMVA).value,3), "MW")
    print("...............................")       
    print("The eigenvalues of matrix W are")
    print(np.round(np.linalg.eig(W.value)[0],2))



ppc = loadcase(case())     #Load test system 
baseMVA=ppc['baseMVA']
Pload=ppc['bus'][:,2]
Qload=ppc['bus'][:,3]
nbus=ppc['bus'].shape[0]              #number of buses
ngen=ppc['gen'].shape[0]
ngenInfo=ppc['gen'].shape[1]
genMatrix=np.zeros((nbus,ngenInfo))
for i in range(ngen):
    genMatrix[int(ppc['gen'][i][0])-1,:]=ppc['gen'][i]
genBusIndex=ppc['gen'][:,0].astype(int)-1
u=np.zeros(nbus, dtype=int)
for i in genBusIndex:
    u[i]=1
# u[genBusIndex[0]]=0
# u[genBusIndex[1]]=0
# u[genBusIndex[2]]=0
# u[genBusIndex[3]]=0
# u[genBusIndex[4]]=0
# u[genBusIndex[5]]=0
OPF(ppc,Pload,Qload,u)