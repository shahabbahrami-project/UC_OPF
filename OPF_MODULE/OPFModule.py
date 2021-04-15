# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:41:43 2021

@author: bahramis
"""

from pypower.api import ppoption, runpf, runopf, opf, printpf
from pypower.makeYbus import makeYbus 
from pypower.ext2int import ext2int 
from pypower.loadcase import loadcase 
from pypower.api import case14 as case
import numpy as np
import json
from OPF_Fast import OPFAC

ppc = loadcase(case())                #Load test system
baseMVA=ppc['baseMVA']
Pload=ppc['bus'][:,2]
Qload=ppc['bus'][:,3]
nbus=ppc['bus'].shape[0]              #number of buses
ngen=ppc['gen'].shape[0]
ngenInfo=ppc['gen'].shape[1]
u=np.zeros(ngen, dtype=int)
for i in range(ngen):
    u[i]=1
weight=10000000

OPFAC(ppc, Pload, Qload, u, weight)