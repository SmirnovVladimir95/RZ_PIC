#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:51:54 2019

@author: vladimirsmirnov
"""

import numba
import numpy as np
from numba import prange


@numba.njit(cache=True, fastmath=True)
def charge_interpolation_numba(rho_grid, pos_z, pos_r, dz, dr, Ntot):
    for i in prange(Ntot):
        cell_z = int(pos_z[i]/dz)
        cell_r = int(pos_r[i]/dr)
        
        #cell_z = int(pos[i*pos_dim + 0]/dz)
        #cell_r = int(pos[i*pos_dim + 1]/dr)
        #hz = (pos[i*pos_dim + 0] - cell_z*dz) / dz
        #hr = (pos[i*pos_dim + 1] - cell_r*dr) / dr
        
        hz = (pos_z[i] - cell_z*dz) / dz
        hr = (pos_r[i] - cell_r*dr) / dr
        rho_grid[cell_z, cell_r] += (1 - hz)*(1 - hr)
        rho_grid[cell_z+1, cell_r] += (1 - hr)*hz
        rho_grid[cell_z+1, cell_r+1] += hz*hr
        rho_grid[cell_z, cell_r+1] += (1 - hz)*hr
        #rho_grid[cell_z*Nr + cell_r] += (1 - hz)*(1 - hr)
        #rho_grid[(cell_z+1)*Nr + cell_r] += (1 - hr)*hz
        #rho_grid[(cell_z+1)*Nr + cell_r+1] += hz*hr
        #rho_grid[cell_z*Nr + cell_r+1] += (1 - hz)*hr
    return rho_grid 

@numba.njit(cache=True, fastmath=True)
def E_interpolation_numba(E, Ez_interp, Er_interp, pos_z, pos_r, dz, dr, Ntot):
    for i in prange(Ntot):
        cell_z = int(pos_z[i]/dz)
        cell_r = int(pos_r[i]/dr)
        hz = (pos_z[i] - cell_z*dz) / dz
        hr = (pos_r[i] - cell_r*dr) / dr
        Ez_interp[i] += E[0][cell_z, cell_r] * (1 - hz) * (1 - hr)
        Ez_interp[i] += E[0][cell_z+1, cell_r] * (1 - hr) * hz
        Ez_interp[i] += E[0][cell_z+1, cell_r+1] * hz * hr
        Ez_interp[i] += E[0][cell_z, cell_r+1] * (1 - hz) * hr
        Er_interp[i] += E[1][cell_z, cell_r] * (1 - hz) * (1 - hr)
        Er_interp[i] += E[1][cell_z+1, cell_r] * (1 - hr) * hz
        Er_interp[i] += E[1][cell_z+1, cell_r+1] * hz * hr
        Er_interp[i] += E[1][cell_z, cell_r+1] * (1 - hz) * hr
    return Ez_interp, Er_interp

@numba.njit(cache=True, fastmath=True)
def B_interpolation_numba(B, Bz_interp, Br_interp, pos_z, pos_r, dz, dr, Ntot):
    for i in prange(Ntot):
        cell_z = int(pos_z[i]/dz)
        cell_r = int(pos_r[i]/dr)
        hz = (pos_z[i] - cell_z*dz) / dz
        hr = (pos_r[i] - cell_r*dr) / dr
        Bz_interp[i] += B[0][cell_z, cell_r] * (1 - hz) * (1 - hr)
        Bz_interp[i] += B[0][cell_z+1, cell_r] * (1 - hr) * hz
        Bz_interp[i] += B[0][cell_z+1, cell_r+1] * hz * hr
        Bz_interp[i] += B[0][cell_z, cell_r+1] * (1 - hz) * hr
        Br_interp[i] += B[1][cell_z, cell_r] * (1 - hz) * (1 - hr)
        Br_interp[i] += B[1][cell_z+1, cell_r] * (1 - hr) * hz
        Br_interp[i] += B[1][cell_z+1, cell_r+1] * hz * hr
        Br_interp[i] += B[1][cell_z, cell_r+1] * (1 - hz) * hr
    return Bz_interp, Br_interp

@numba.njit(cache=True, fastmath=True)
def method_node_volume_numba(dz, dr, node_vol):
    nz = node_vol.shape[0]
    nr = node_vol.shape[0]
    for i in range(0,nz):
        for j in range(0,nr):
            j_min = j - 0.5
            j_max = j + 0.5
            if (j_min < 0): j_min = 0
            if (j_max > nr - 1): j_max = nr - 1
            a = 0.5 if (i == 0 or i == nz - 1) else 1.0
            #note, this is r*dr for non-boundary nodes
            node_vol[i][j] = a*dz*((j_max*dr)**2 - (j_min*dr)**2)*np.pi
    return node_vol

import time

if __name__ == '__main__':
    rho_grid = np.random.random(size=(100, 50))
    pos = np.random.random(size=int(3e6))
    dz = dr = 0.1
    Ntot = int(1e6)
    pos_dim = 3
    #Nz = 100
    #Nr = 50
    #rho_grid = rho_grid.reshape(Nz*Nr)
    Ez = np.random.random(size=(100, 50))
    Er = np.random.random(size=(100, 50))
    Ez_interp = np.random.random(size=pos.shape[0])
    Er_interp = np.random.random(size=pos.shape[0])
    t0 = time.time()
    #a = charge_interpolation_numba(rho_grid, pos, dz, dr, Ntot, pos_dim)
    #E_interpolation_numba(Ez, Er, Ez_interp, Er_interp, pos, dz, dr)
    print time.time() - t0
