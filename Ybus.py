# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:14:19 2021

@author: bahramis
"""


from pypower.api import case14, ppoption, runpf, printpf
from pypower.makeYbus import makeYbus 
from pypower.ext2int import ext2int 
from pypower.loadcase import loadcase 

def Ybus(ppc):
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    return Ybus