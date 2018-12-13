#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:03:48 2018

@author: jeguerra
"""

import datetime as dt  # Python standard library datetime module
import numpy as np
import math as mt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mpl_toolkits

def cheblb(REFS):
       # Compute Chebyshev CGL nodes and weights
       ep = REFS[2] - 1
       xc = np.array(range(REFS[2]))
       xi = np.cos(mt.pi / ep * xc)
       
       w = np.ones(REFS[2])
       w[0] = 0.5
       w[ep] = 0.5
       
       # return column vector grid and weights
       xi = (np.mat(xi)).T
       w = (np.mat(w)).T
       
       return xi, w

def computeBackground(REFS):
       
       # Initialize and make column vectors
       thetaBar = np.mat(np.zeros(REFS[2]))
       thetaBar = thetaBar.T
       rhoBar = np.mat(np.zeros(REFS[2]))
       rhoBar = rhoBar.T
       
       return thetaBar, rhoBar

def computeGridDerivatives(REFS):
       
       # Initialize grid and make column vector
       zc, w = cheblb(REFS)
       zg = 0.5 * REFS[0] * (1.0 - zc) 
       
       # Initialize derivative matrix objects
       DDZ = np.mat(np.eye(REFS[2]))
       DDZ2 = np.mat(np.eye(REFS[2]))
       
       return zg, DDZ, DDZ2

if __name__ == '__main__':
       
       # Set up the grid using Tempest nominal HS data near the equator
       zH = 30000.0
       zTP = 16000.0
       NZ = 50
       T0 = 295.0
       GamTrop = 1.9E-3 # K per meter
       GamStrt = 2.4E-2 # K per meter
       
       # Put all the input parameters into a list REFS
       REFS = [zH, zTP, NZ, T0, GamTrop, GamStrt]
       
       # Compute the grid and derivative matrices
       zg, DDZ, DDZ2 = computeGridDerivatives(REFS)
       
       # Compute the background profiles (theta and rho) based on two lapse rates in theta
       thetaBar, rhoBar = computeBackground(REFS)