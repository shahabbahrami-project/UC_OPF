# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:07:49 2021

@author: bahramis
"""


from pypower.api import ppoption, runpf, runopf, opf, printpf
from pypower.makeYbus import makeYbus 
from pypower.ext2int import ext2int 
from pypower.loadcase import loadcase 
from pypower.api import case300 as case
import numpy as np
import json

def OPFAC(ppc, Pload, Qload, u, weight):
    baseMVA, bus, gen, branch = ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    nbus=ppc['bus'].shape[0]
    ngen=ppc['gen'].shape[0]
    ngenInfo=ppc['gen'].shape[1]
    
    ppc["bus"][:,2]=Pload
    ppc["bus"][:,3]=Qload
    for i in range(ngen):
        ppc["gen"][i,3]=u[i]*ppc["gen"][i,3]
        ppc["gen"][i,4]=u[i]*ppc["gen"][i,4]
        ppc["gen"][i,8]=u[i]*ppc["gen"][i,8]
        ppc["gen"][i,9]=u[i]*ppc["gen"][i,9]  

    DeltaPMatrix=np.zeros((nbus,ngenInfo))
    for i in range(nbus):
        DeltaPMatrix[i,:]=[   i+1,    0.,    0.,  0., 0,    1.,  100.,    1.,  1000,
                              -1000.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
                              0.,    0.,    0.] 
        
    genNew=np.append(gen, DeltaPMatrix, axis=0)
    ppc["gen"]=genNew 
    DeltaPCost=np.zeros((nbus,ppc["gencost"].shape[1]))
    DeltaPCost[:,-3]=weight
    DeltaPCost[:,-4]=3
    DeltaPCost[:,0]=2
    newGenCost=np.append(ppc["gencost"],DeltaPCost , axis=0) 
    ppc["gencost"]=newGenCost
    ppc["bus"][:,1]=2
    ppc["bus"][0,1]=3 
    runopf(ppc)


ppc = loadcase(case())                #Load test system
baseMVA=ppc['baseMVA']
Pload=ppc['bus'][:,2]
Qload=ppc['bus'][:,3]
nbus=ppc['bus'].shape[0]              #number of buses
ngen=ppc['gen'].shape[0]
ngenInfo=ppc['gen'].shape[1]
# genMatrix=np.zeros((nbus,ngenInfo))
# for i in range(ngen):
#     genMatrix[int(ppc['gen'][i][0])-1,:]=ppc['gen'][i]
# genBusIndex=ppc['gen'][:,0].astype(int)-1
u=np.zeros(ngen, dtype=int)
for i in range(ngen):
    u[i]=1



weight=10000000

OPFAC(ppc, Pload, Qload, u, weight)