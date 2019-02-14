#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:03:48 2018

@author: jeguerra
"""

import datetime as dt  # Python standard library datetime module
import numpy as np
import math as mt
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cheblb(REFS):
       # Compute Chebyshev CGL nodes and weights
       ep = REFS[2] - 1
       xc = np.array(range(REFS[2]))
       xi = np.cos(mt.pi / ep * xc)
       
       w = mt.pi / (ep + 1) * np.ones(REFS[2])
       w[0] *= 0.5
       w[ep] *= 0.5
       
       # return column vector grid and weights
       xi = (np.mat(xi)).T
       w = (np.mat(w)).T
   
       return xi, w
   
def chebpolym(REFS, xi):
       # Compute Chebyshev pols (first kind) into a matrix transformation
       N = REFS[2] - 1
       CTM = np.mat(np.zeros((N+1, N+1)))
       
       CTM[:,0] = np.ones((N+1, 1))
       CTM[:,1] = xi
       
       # 3 Term recursion
       for ii in range(2, N+1):
              CTM[:,ii] = 2.0 * \
              np.multiply(xi, CTM[:,ii-1]) - \
              CTM[:,ii-2]
              
       return CTM

def computeBackground(PHYS, REFS, zg, DDZ):
       N = REFS[2]
       # Initialize and make column vectors
       thetaBar = np.mat(np.zeros(REFS[2]))
       thetaBar = thetaBar.T
       rhoBar = np.mat(np.zeros(REFS[2]))
       rhoBar = rhoBar.T
       pBar = np.mat(np.zeros(REFS[2]))
       pBar = pBar.T
    
       # Set the 2 linear branches of theta
       thetaTRP = REFS[3] + REFS[4] * zg
       thetaTPP = REFS[3] + REFS[4] * REFS[1]
       thetaSTR = thetaTPP + REFS[5] * (zg - REFS[1])
    
       thetaBar = thetaTRP
       for ii in range(N):
              if zg[ii] >= REFS[1]:
                     thetaBar[ii] = thetaSTR[ii]
            
       # Compute the Neumann boundary value at the top z = H
       A = - PHYS[0] * (PHYS[1]**PHYS[4]) / PHYS[3]
       thetaBarI = np.reciprocal(thetaBar)
       #rho0 = PHYS[0] / (PHYS[3] * REFS[3])
       #dPdzH = - A * thetaBarI[N-1] * (PHYS[1]**(1.0 - PHYS[4]) + \
       #          rho0 * PHYS[0] * REFS[0] * (PHYS[4] - 1.0) * PHYS[1]**(-PHYS[4]))
    
       # Solve the system... Dirichlet condition at z = 0m (SEE NOTES)
       pBar[range(1,N)] = A * PHYS[4] * \
                              la.solve(DDZ[1:N,1:N], thetaBarI[1:N])
       pBar += PHYS[1]**PHYS[4]
       pBar = np.power(pBar, 1.0 / PHYS[4])
                              
       # Recover temperature and density
       TBar = np.multiply(thetaBar, np.power(1.0 / PHYS[1] * pBar, PHYS[4]))
       rhoBar = np.multiply(pBar, np.reciprocal(TBar))
       rhoBar *= 1.0 / PHYS[3]
                              
       # Check the background state
       '''
       fig, axes = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
       axes[0,0].plot(zg, thetaBar)
       axes[0,1].plot(zg, pBar)
       axes[1,0].plot(zg, TBar)
       axes[1,1].plot(zg, rhoBar)
       plt.show()
       '''
       return thetaBar, rhoBar, pBar

def computeGridDerivativesZ(REFS):
    
       NZ = REFS[2]
       # Initialize grid and make column vector
       zc, w = cheblb(REFS)
       zg = 0.5 * REFS[0] * (1.0 - zc) 
   
       # Get the Chebyshev transformation matrix
       CTD = chebpolym(REFS, zc)
   
       # Make the weights into a diagonal matrix
       W = np.eye(NZ)
       for ii in range(NZ):
              W[ii,ii] = w[ii]
   
       # Compute scaling for the forward transform
       S = np.eye(NZ)
   
       for ii in range(NZ - 1):
              S[ii,ii] = ((CTD[:,ii]).T * W * CTD[:,ii]) ** (-1)

       S[NZ-1,NZ-1] = 1.0 / mt.pi
   
       # Compute the spectral derivative coefficients
       SDIFF = np.zeros((NZ,NZ))
       SDIFF[NZ-2,NZ-1] = 2.0 * NZ
   
       for ii in reversed(range(NZ - 2)):
              A = 2.0 * (ii + 1)
              B = 1.0
              if ii > 0:
                     c = 1.0
              else:
                     c = 2.0
            
              SDIFF[ii,:] = B / c * SDIFF[ii+2,:]
              SDIFF[ii,ii+1] = A / c
    
       # Chebyshev spectral transform in matrix form
       STR_L = S * CTD * W;
       # Chebyshev spatial derivative based on spectral differentiation
       # Domain scale factor included here
       DDZ = - (2.0 / REFS[0]) * CTD.T * SDIFF * STR_L;
       # Compute 2nd derivative
       SDIFF2 = np.matmul(SDIFF, SDIFF)
       DDZ2 = - (2.0 / REFS[0]) * CTD.T * SDIFF2 * STR_L;
       #DDZ2 = np.matmul(DDZ, DDZ)
       
       # Make a test function and its derivative (DEBUG)
       """
       zv = (1.0 / zH) * zg
       zv2 = np.multiply(zv, zv)
       Y = 4.0 * np.exp(-2.0 * zv) + \
       np.cos(4.0 * mt.pi * zv2);
       DY = -8.0 * np.exp(-2.0 * zv)
       term1 = 8.0 * mt.pi * zv
       term2 = np.sin(4.0 * mt.pi * zv2)
       DY -= np.multiply(term1, term2);
    
       DYD = np.matmul(DDZ, Y)
       plt.figure
       plt.plot(zv, Y, zv, DY, zv, DYD)
       """
       
       return zg, CTD, DDZ, DDZ2

def computeGridDerivativesP(PHYS, REFS, pBar):
    
       NZ = REFS[2]
       # Initialize grid and make column vector
       zc, w = cheblb(REFS)
       DP = (pBar[NZ-1] - pBar[0]).flat[0]
       p0pH = (pBar[NZ-1] + pBar[0]).flat[0]
       pg = np.flip(0.5 * np.add(DP * zc, p0pH), 0) 
   
       # Get the Chebyshev transformation matrix
       CTD = chebpolym(REFS, zc)
   
       # Make the weights into a diagonal matrix
       W = np.eye(NZ)
       for ii in range(NZ):
              W[ii,ii] = w[ii]
   
       # Compute scaling for the forward transform
       S = np.eye(NZ)
   
       for ii in range(NZ - 1):
              S[ii,ii] = ((CTD[:,ii]).T * W * CTD[:,ii]) ** (-1)

       S[NZ-1,NZ-1] = 1.0 / mt.pi
   
       # Compute the spectral derivative coefficients
       SDIFF = np.zeros((NZ,NZ))
       SDIFF[NZ-2,NZ-1] = 2.0 * NZ
   
       for ii in reversed(range(NZ - 2)):
              A = 2.0 * (ii + 1)
              B = 1.0
              if ii > 0:
                     c = 1.0
              else:
                     c = 2.0
            
              SDIFF[ii,:] = B / c * SDIFF[ii+2,:]
              SDIFF[ii,ii+1] = A / c
    
       # Chebyshev spectral transform in matrix form
       STR_L = S * CTD * W;
       # Chebyshev spatial derivative based on spectral differentiation
       # Domain scale factor included here
       DDP = - (2.0 / DP) * CTD.T * SDIFF * STR_L;
       # Compute 2nd derivative
       SDIFF2 = np.matmul(SDIFF, SDIFF)
       DDP2 = - (2.0 / DP) * CTD.T * SDIFF2 * STR_L;
       #DDP2 = np.matmul(DDP, DDP)
   
       return pg, DDP, DDP2

if __name__ == '__main__':
       
       # Set physical constants
       gc = 9.80601
       P0 = 1.0E5
       cp = 1004.5
       Rd = 287.06
       Kp = Rd / cp
    
       # Put all the physical constants into a list PHYS
       PHYS = [gc, P0, cp, Rd, Kp]
       
       # Set up the grid using Tempest nominal HS data near the equator
       zH = 30000.0
       zTP = 16000.0
       NZ = 256
       T0 = 295.0
       GamTrop = 1.9E-3 # K per meter
       GamStrt = 2.4E-2 # K per meter
   
       # Put all the input parameters into a list REFS
       REFS = [zH, zTP, NZ, T0, GamTrop, GamStrt]
       
       # Compute the geometric grid and derivative matrices
       zg, CTD, DDZ, DDZ2 = computeGridDerivativesZ(REFS)
    
       # Compute the background profiles (theta and rho) based on two lapse rates in theta
       thetaBar, rhoBar, pBar = computeBackground(PHYS, REFS, zg, DDZ)
       
       # Compute the isobaric grid and derivative matrices
       pg, DDP, DDP2 = computeGridDerivativesP(PHYS, REFS, pBar)
       
       # Compute the variable stratification and make a diagonal matrix
       c = 61.5
       rhoBar2 = np.power(rhoBar, 2.0)
       rhoBar2I = np.reciprocal(rhoBar2)
       thetaBarI = np.reciprocal(thetaBar)
       NBV = np.matmul(DDZ, thetaBar)
       NBV = np.multiply(NBV, thetaBarI)
       G2 = 1.0 / gc * np.multiply(rhoBar2I, NBV)
       G2M = np.eye(NZ)
       for ii in range(NZ):
              G2M[ii,ii] = G2[ii]
              
       # Compute the operator
       EOP = DDP2 + G2M
       
       # Apply Dirichlet BC
       EOPS = EOP[1:NZ-1,1:NZ-1]
       
       # Compute eigensolve
       ew, ev = np.linalg.eig(EOP)
       
       # Sort the eigenvalues and vectors ascending
       sdex = np.argsort(ew)
       lam = ew[sdex]
       Psi = ev[:,sdex]
       
       #%% Plot the first eigenvector
       plt.plot(zg, Psi[:,10])