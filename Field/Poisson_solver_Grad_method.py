#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:43:11 2019

@author: vladimirsmirnov
"""
import numpy as np
from scipy.optimize import newton_krylov
import seaborn as sns

def Poisson_solver_GRD_METHOD(phi_init, b, CathodeV, AnodeV, Nz, Nr, dz, dr, CathodeR, tol):
    nx, ny, hx, hy = Nz, Nr, dz, dr
    P_left, P_right = CathodeV, CathodeV
    P_top = AnodeV
    R = lambda j: j*hy
    P = phi_init
    r = np.zeros_like(P)
    for i in range(nx):
        for j in range(ny):
            r[i][j] = R(j)
    def residual(P):
        d2x = np.zeros_like(P)
        d2y = np.zeros_like(P)
        dy2 = np.zeros_like(P)
    
        d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
        d2x[0,:CathodeR]    = (P[1,:CathodeR]    - 2*P[0,:CathodeR]    + P_left)/hx/hx
        d2x[-1,:CathodeR]   = (P_right - 2*P[-1,:CathodeR]   + P[-2,:CathodeR])/hx/hx
        d2x[0,CathodeR:]    = (P[1,CathodeR:]    - 2*P[0,CathodeR:]    + P[0,CathodeR:])/hx/hx
        d2x[-1,CathodeR:]    = (P[-1,CathodeR:]    - 2*P[-1,CathodeR:]    + P[-2,CathodeR:])/hx/hx
        
        d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
        d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P[:,0])/hy/hy
        d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy
        
        dy2[:,1:-1] = (P[:,2:] - P[:,:-2])/(2*hy)/r[:,1:-1]
        dy2[:,0] = (P[:,1]  - P[:,0])/(2.*hy)/r[:,1]
        dy2[:,-1] = (P_top  - P[:,-2])/(2.*hy)/r[:,-1]
        return d2x + d2y + dy2 + b
    
    # solve
    guess = phi_init
    solution = newton_krylov(residual, guess, f_tol=tol)
    return solution

if __name__ == '__main__':
    
    sol = Poisson_solver_GRD_METHOD(phi_init=np.ones((100, 50))*(-50.), b=np.ones((100, 50))*(0), CathodeV=-100., AnodeV=0., 
                                    Nz=100, Nr=50, dz=2e-3, dr=2e-3, CathodeR=25, tol=1e-4)
    sns.heatmap(sol.T)