#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:15:34 2019

@author: vladimirsmirnov
"""

import numpy as np
import numba

@numba.njit(cache=True)
def gauss_seidel_integrator(g, phi, b, r, dr, dr2, dz, dz2):
    for i in range(1, phi.shape[0]-1, 1):
        for j in range(1, phi.shape[1]-1, 1):
            g[i,j] = (b[i, j] + (g[i,j-1]+phi[i,j+1])/dr2 +
                         (phi[i,j+1]-g[i,j-1])/(2*dr*r[i,j]) +
                         (g[i-1,j] + phi[i+1,j])/dz2) / (2/dr2 + 2/dz2)
    return g

@numba.jit(cache=True, target='cpu')
def finite_diffence_step(b, phi, r, dz2, dr2, dr):
    g = (b + (phi[2] + phi[3])/dr2 + (phi[2] - phi[3])/(2*dr*r) + 
                        (phi[0] + phi[1])/dz2)/(2/dr2 + 2/dz2)
    return g
    
# phi = [phi[i-1, j], phi[i+1, j], phi[i,j-1], phi[i,j+1]]
@numba.jit(cache=False, parallel=False)
def finite_diffence_scheme(phi, b, r, g, nz, nr, dz2, dr2, dr):
    for i in numba.prange(1, nz-1, 1):
        for j in xrange(1, nr-1, 1):
            phi_current = np.array([phi[i-1, j], phi[i+1, j], phi[i,j-1], phi[i,j+1]])
            g[i, j] = finite_diffence_step(b=b[i, j], phi=phi_current, 
                             r=r[i,j], dz2=dz2, dr2=dr2, dr=dr)
            
            #g[i,j] = (b[i][j] + (phi[i,j-1]+phi[i,j+1])/dr2 +
            #             (phi[i,j-1]-phi[i,j+1])/(2*dr*r[i,j]) +
            #             (phi[i-1,j] + phi[i+1,j])/dz2) / (2/dr2 + 2/dz2)
            

if __name__ == '__main__':
    pass
        