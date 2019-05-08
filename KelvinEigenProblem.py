#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:03:48 2018

@author: jeguerra
"""

import numpy as np
from scipy import linalg as las
from scipy.optimize import curve_fit
import math as mt
from numpy import linalg as lan
import matplotlib.pyplot as plt

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
       
def computeAdjustedOperatorNBC(D2A, DOG, DD, tdex, isGivenValue, DP):
       # D2A is the operator to adjust
       # DOG is the original operator to adjust (unadjusted)
       # DD is the 1st derivative operator
       DOP = np.zeros(DD.shape)
       # Get the column span size
       NZ = DD.shape[1]
       cdex = range(NZ)
       cdex = np.delete(cdex, tdex)
       
       # For prescribed value:
       if isGivenValue:
              scale = - DD[tdex,tdex]
       # For matching at infinity
       else:
              scale = (DP - DD[tdex,tdex])
              
       # Loop over columns of the operator and adjust for BC at z = H (tdex)
       for jj in cdex:
              factor = DD[tdex,jj] / scale
              v1 = (D2A[:,jj]).flatten()
              v2 = (DOG[:,tdex]).flatten()
              nvector = v1 + factor * v2
              DOP[:,jj] = nvector
       
       return DOP

def computeBackground(PHYS, REFS, zg, DDZ):
       N = REFS[2]
       # Initialize and make column vectors
       thetaBar = np.mat(np.zeros(REFS[2]))
       thetaBar = thetaBar.T
       dThdZ = np.mat(np.zeros(REFS[2]))
       dThdZ = dThdZ.T
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
                     dThdZ[ii] = REFS[5]
              else:
                     dThdZ[ii] = REFS[4]
            
       # Compute the Neumann boundary value at the top z = H
       A = - PHYS[0] * (PHYS[1]**PHYS[4]) / PHYS[3]
       thetaBarI = np.reciprocal(thetaBar)
       
       #%% Impose BC p^K = 0 @ z = 0 and d(p^K)dz = B @ z = H matched to p^K = 0 at Inf
       # Specify the derivative at the model top
       dpdZ_H = PHYS[4] * A * thetaBarI[N-1]
       dpdZ_H = dpdZ_H[0,0]
       # Compute adjustment to the derivative matrix operator
       DOP = computeAdjustedOperatorNBC(DDZ, DDZ, DDZ, N-1, True, pBar)

       # Impose resulting Dirichlet conditions p^K top and bottom
       NE = N-1
       DOPS = DOP[1:NE,1:NE]
       
       # Index of interior nodes
       idex = range(1,NE)
       # Index of left and interior nodes
       bdex = range(0,NE)
       
       # Compute the forcing due to matching at the model top
       f = -dpdZ_H / DDZ[N-1,N-1] * DDZ[:,N-1]
       F = np.add(thetaBarI, f)
       # Solve the system for p^K
       pBar[idex] = A * PHYS[4] * lan.solve(DOPS, F[1:NE])
       
       # Compute and set the value at the top that satisfies the BC
       dPdZ_partial = np.dot(DDZ[N-1,bdex], pBar[bdex])
       pBar[N-1] = (dpdZ_H - dPdZ_partial) / DDZ[N-1,N-1]
                              
       #%% Reconstruct hydrostatic pressure p from p^K
       pBar[0:N] += PHYS[1]**PHYS[4]
       pBar = np.power(pBar, 1.0 / PHYS[4])
                              
       #%% Recover temperature and density
       TBar = np.multiply(thetaBar, np.power(1.0 / PHYS[1] * pBar, PHYS[4]))
       rhoBar = np.multiply(pBar, np.reciprocal(TBar))
       rhoBar *= 1.0 / PHYS[3]
                              
       #%% Check the background state
       '''
       fig, axes = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
       axes[0,0].plot(zg, thetaBar, 'k-', linewidth=2)
       axes[0,0].set_title('$\\theta(z)$ $K$')
       axes[0,0].grid(b=True, which='both', axis='both')
       axes[0,1].plot(zg, pBar, linewidth=2)
       axes[0,1].set_title('$p(z)$ $Pa$')
       axes[0,1].grid(b=True, which='both', axis='both')
       axes[1,0].plot(zg, TBar, linewidth=2)
       axes[1,0].set_title('$T(z)$ $K$')
       axes[1,0].set_xlabel('z (m)')
       axes[1,0].grid(b=True, which='both', axis='both')
       axes[1,1].plot(zg, rhoBar, linewidth=2)
       axes[1,1].set_title('$\\rho(z)$ $kgm^{-3}$')
       axes[1,1].set_xlabel('z (m)')
       axes[1,1].grid(b=True, which='both', axis='both')
       plt.show()
       plt.savefig("2LAYER_stratification.png")
       #'''
       return thetaBar, dThdZ, rhoBar, pBar

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
       
       return zg, SDIFF, DDZ, DDZ2

def computeGridDerivativesP(PHYS, REFS, pBar, dpdz, DDZ):
    
       NZ = REFS[2]
       dpdzI = np.reciprocal(dpdz)
   
       # Make the weights into a diagonal matrix
       IDPDZ = np.eye(NZ)
       for ii in range(NZ):
              IDPDZ[ii,ii] = dpdzI[ii]
              
       # Chebyshev spatial derivative based on spectral differentiation
       # Computed from the Chain Rule on DDZ
       DDP = IDPDZ * DDZ;
       # Compute 2nd derivative in pressure
       DDP2 = DDP * DDP
   
       return pBar, DDP, DDP2

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
       zH = 35000.0
       zTP = 16000.0
       NZ = 512
       T0 = 295.0
       GamTrop = 1.88E-3 # K per meter
       GamStrt = 2.37E-2 # K per meter
   
       # Put all the input parameters into a list REFS
       REFS = [zH, zTP, NZ, T0, GamTrop, GamStrt]
       
       # Compute the geometric grid and derivative matrices
       zg, SDIFFZ, DDZ, DDZ2 = computeGridDerivativesZ(REFS)
    
       # Compute the background profiles (theta and rho) based on two lapse rates in theta
       thetaBar, dThdZ, rhoBar, pBar = computeBackground(PHYS, REFS, zg, DDZ)
       
       # Compute the isobaric grid and derivative matrices
       pg, DDP, DDP2 = computeGridDerivativesP(PHYS, REFS, pBar, -gc * rhoBar, DDZ)
       
       # Compute the variable stratification and make a diagonal matrix
       dpdz = -gc * rhoBar
       dpdzI = np.reciprocal(dpdz)
       ddpdz2I = np.matmul(DDZ, dpdzI)
       
       def quadPol(x, a, b, c):
              return a * np.power(x,2.0) + b * x + c
       
       def cubicPol(x, a, b, c, d):
              return a * np.power(x,3.0) + b * np.power(x,2.0) + c * x + d
       
       def decayFunc(x, a, b):
              return a * np.reciprocal(np.power(x,b))
       
       # Compute rho as a function of pressure
       popt, covt = curve_fit(cubicPol, np.ravel(pBar), np.ravel(rhoBar))
       rhoBarP = cubicPol(pBar, *popt)
       
       # Compute the Brunt-Vaisala frequency as a function of z
       NBV = dThdZ
       thetaBarI = np.reciprocal(thetaBar)
       NBV = np.multiply(NBV, thetaBarI)
       
       # Compute fit of Brunt-Vaisala as a function of pressure (2 branches)
       zdex1 = np.where(zg >= zTP)
       zdex2 = np.where(zg < zTP)
       # Evaluate the branches, force endpoints
       sigma = np.ones(len(zdex1[0]))
       sigma[[0, -1]] = 1.0E-3
       popt1, covt1 = curve_fit(quadPol, np.ravel(pBar[zdex1]), np.ravel(NBV[zdex1]), sigma=sigma)
       NBV1 = quadPol(pBar[zdex1], *popt1)
       popt1, covt1 = curve_fit(quadPol, np.ravel(pBar[zdex1]), np.ravel(thetaBar[zdex1]), sigma=sigma)
       thetaBar1 = quadPol(pBar[zdex1], *popt1)
       sigma = np.ones(len(zdex2[0]))
       sigma[[0, -1]] = 1.0E-3
       popt2, covt2 = curve_fit(cubicPol, np.ravel(pBar[zdex2]), np.ravel(NBV[zdex2]), sigma=sigma)
       NBV2 = cubicPol(pBar[zdex2], *popt2)
       popt2, covt2 = curve_fit(cubicPol, np.ravel(pBar[zdex2]), np.ravel(thetaBar[zdex2]), sigma=sigma)
       thetaBar2 = cubicPol(pBar[zdex2], *popt2)
       # Compute the full column stratification
       NBVP = np.concatenate((NBV2.T, NBV1.T))
       thetaBarP = np.concatenate((thetaBar2.T, thetaBar1.T))
       
       rhoBar2P = np.power(rhoBarP, 2.0)
       rhoBar2IP = np.reciprocal(rhoBar2P)
       G2 = 1.0 / gc * np.multiply(rhoBar2IP, NBVP)
       
       IG2 = np.reciprocal(G2)
       G2M = np.eye(NZ)
       for ii in range(NZ):
              G2M[ii,ii] = G2[ii]
              
       RP = 1.0
       LHSOP = computeAdjustedOperatorNBC(DDP2, DDP2, DDP, NZ-1, False, RP)
       RHSOP = computeAdjustedOperatorNBC(G2M, G2M, DDP, NZ-1, False, RP)
              
       # Apply Dirichlet BC @ z = 0 and z = H
       NE = NZ-1
       LHSOPS = LHSOP[1:NE,1:NE]
       RHSOPS = RHSOP[1:NE,1:NE]
       
       # Compute eigensolve
       ew, ev = las.eig(LHSOPS, b=-RHSOPS, left=False, right=True)
       
       # Recover the Neumann BC values (TOP OF ATMOSPHERE)
       scale = RP - DDP[NE,NE]
       BCeq = 1.0 / scale * DDP[NE,1:NE]
       evTop = np.matmul(BCeq, ev)
       
       #%% Sort the eigenvalues and vectors ascending
       Psi = np.zeros((NZ,1))
       sdex = np.argsort(np.real(ew))
       lam = ew[sdex]
       ev = ev[:,sdex]
       evTop = (evTop.T)[sdex]
       
       # Find the next to smallest eigenvalue (n = 1)
       vdex = np.argmin(np.abs(lam))
       vdex += 1
       Psi[1:NE,0] = (ev[:,vdex-1]).flatten()
       Psi[NE,0] = (evTop[vdex-1]).flatten()
       c1 = mt.sqrt(1.0 / abs(lam[vdex]))
       
       #%% Plot the first eigenvector
       fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6), tight_layout=True)
       ax0.plot(1.0E-2 * pg, Psi, 'k', label='$\psi(z)$, c = %5.3f $ms^{-1}$' % c1)
       ax0.set_xlabel('p (hPa)')
       ax0.set_ylabel('Eigenfunction')
       ax0.set_title('Kelvin Wave Vertical Structure')
       ax0.legend()
       ax0.grid(b=True, which='both', axis='both')
       ax0.invert_xaxis()
       
       ax1.plot(1.0E-3 * zg, Psi, 'k', label='$\psi(z)$, c = %5.3f $ms^{-1}$' % c1)
       ax1.set_xlabel('z (km)')
       ax1.set_ylabel('Eigenfunction')
       ax1.set_title('Kelvin Wave Vertical Structure')
       ax1.legend()
       ax1.grid(b=True, which='both', axis='both')
       plt.show()
       
       plt.savefig("KelvinWaveStructure.png")
       
       #%% Recover the physical fields
       cv = c1
       RT = np.multiply(thetaBar, rhoBar)
       IRT = np.reciprocal(RT)
       rhoBarI = np.reciprocal(rhoBar)
       wv = np.multiply(-1.0 / gc * rhoBarI, Psi)
       uv = -np.matmul(DDP, Psi)
       ExnerP = cv * uv
       BUO = np.matmul(DDP, ExnerP)
       BUO[0] = lam[vdex] * G2[0] * Psi[0]
       BUO[-1] = lam[vdex] * G2[-1] * Psi[-1]
       theta = np.multiply(-RT, BUO) 
       
       fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), tight_layout=True)
       
       ax0.plot(1.0E-3 * zg, P0 * ExnerP, 'k', label='$\psi(z)$, c = %5.3f $ms^{-1}$' % cv)
       ax0.set_xlabel('$z (km)$')
       ax0.set_ylabel('$p \: \prime \: (Pa)$')
       ax0.set_title('Kelvin Wave Vertical Structure: Pressure')
       ax0.legend()
       ax0.grid(b=True, which='both', axis='both')
       
       ax1.plot(1.0E-3 * zg, theta, 'k', label='$\psi(z)$, c = %5.3f $ms^{-1}$' % cv)
       ax1.set_xlabel('$z (km)$')
       ax1.set_ylabel('$\\theta \: \prime \: (K)$')
       ax1.set_title('Kelvin Wave Vertical Structure: Potential Temperature')
       ax1.legend()
       ax1.grid(b=True, which='both', axis='both')
       
       ax2.plot(1.0E-3 * zg, uv, 'k', label='$\psi(z)$, c = %5.3f $ms^{-1}$' % cv)
       ax2.set_xlabel('$z (km)$')
       ax2.set_ylabel('$u \: \prime \: (ms^{-1})$')
       ax2.set_title('Kelvin Wave Vertical Structure: Horizontal Velocity')
       ax2.legend()
       ax2.grid(b=True, which='both', axis='both')
       
       ax3.plot(1.0E-3 * zg, wv, 'k', label='$\psi(z)$, c = %5.3f $ms^{-1}$' % cv)
       ax3.set_xlabel('$z (km)$')
       ax3.set_ylabel('$w \: \prime \: (ms^{-1})$')
       ax3.set_title('Kelvin Wave Vertical Structure: Vertical Velocity')
       ax3.legend()
       ax3.grid(b=True, which='both', axis='both')
       plt.show()
       
       plt.savefig("KelvinWaveStructureVariables.png")
       
       #%%
       # Make a test function and its derivative (DEBUG)
       '''
       zv = (1.0 / zH) * zg
       zv2 = np.multiply(zv, zv)
       Y = 4.0 * np.exp(-5.0 * zv) + \
       np.cos(4.0 * mt.pi * zv2);
       DY = -20.0 * np.exp(-5.0 * zv)
       term1 = 8.0 * mt.pi * zv
       term2 = np.sin(4.0 * mt.pi * zv2)
       DY -= np.multiply(term1, term2);
    
       DYD = zH * np.matmul(DDZ, Y)
       plt.figure(figsize=(8, 6), tight_layout=True)
       plt.plot(zv, Y, label='Function')
       plt.plot(zv, DY, 'r-', label='Analytical Derivative')
       plt.plot(zv, DYD, 'k--', label='Spectral Derivative')
       plt.xlabel('Domain')
       plt.ylabel('Function')
       plt.title('Chebyshev Derivative Test')
       plt.grid(b=True, which='both', axis='both')
       plt.legend()
       plt.savefig("DerivativeTest.png")
       '''   