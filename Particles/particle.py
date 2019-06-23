#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:44:46 2019

@author: vladimirsmirnov
"""

import numpy as np
import matplotlib.pyplot as plt
import motion_integrator
from numba_methods import method_node_volume_numba, charge_interpolation_numba
from numba_methods import E_interpolation_numba, B_interpolation_numba
from scipy.constants import k
from NumpyDynamicArray import NumpyDynamicArray1D

class Particle():
    def __init__(self, mass, charge, Nz, Nr, dz, dr, dens = None, PtclperCell = None, single = False, multy_coeff = 5):
        self.dens = dens
        self.PtclperCell = PtclperCell
        self.Nz = Nz
        self.Nr = Nr
        self.dz = dz
        self.dr = dr
        self.single = single
        if self.single == True or dens == None:
            self.Ntot = 1
            self.PtclperMacro = 1
        else:
            self.Ntot = self.Nz*self.Nr*self.PtclperCell
            V = self.Nz*self.dz*np.pi*(self.Nr*self.dr)**(2)
            self.PtclperMacro = self.dens*V/self.Ntot
        self.mass = mass*self.PtclperMacro
        self.charge = charge*self.PtclperMacro
        self.multy_coeff = multy_coeff
        self.obj_pos = [NumpyDynamicArray1D(size = self.Ntot, dtype=float, multy_coeff=self.multy_coeff),
                  NumpyDynamicArray1D(size = self.Ntot, dtype=float, multy_coeff=self.multy_coeff)]
        self.obj_vel = [NumpyDynamicArray1D(size = self.Ntot, dtype=float, multy_coeff=self.multy_coeff),
                  NumpyDynamicArray1D(size = self.Ntot, dtype=float, multy_coeff=self.multy_coeff),
                  NumpyDynamicArray1D(size = self.Ntot, dtype=float, multy_coeff=self.multy_coeff)]
        
        #self.pos = np.zeros((self.Ntot, 3))
        #self.vel = np.zeros((self.Ntot, 3))
        self.internal_ptcls_idx = np.empty(self.Ntot, dtype=bool)
    
    def set_particles_pos(self, dens_func_r = lambda x: x, dens_func_z = lambda x: x, z = None, r = None, seed=None):
        if self.single == False or z == None or r == None:
            np.random.seed(seed=seed)
            #self.pos[:, 0] = (self.Nz-2)*self.dz*dens_func_z(np.random.random(size = self.Ntot)) + self.dz
            #self.pos[:, 1] = (self.Nr-2)*self.dr*dens_func_r(np.random.random(size = self.Ntot)) + self.dz
            self.obj_pos[0].array_set((self.Nz-2)*self.dz*dens_func_z(np.random.random(size = self.Ntot)) + self.dz)
            self.obj_pos[1].array_set((self.Nr-2)*self.dr*dens_func_r(np.random.random(size = self.Ntot)) + self.dz)
        else:
            assert z != None and r != None, "set z, r coordinates"
            #self.pos[:, 0] = z
            #self.pos[:, 1] = r
            self.obj_pos[0].array_set(value=z, idx=0)
            self.obj_pos[1].array_set(value=r, idx=0)
        
    def get_particles_pos(self, ptcl_idx=None):
        if self.Ntot == 0:
            print "No particles in the system"
            return
        if ptcl_idx is None:
            return [self.obj_pos[0].array_get(), self.obj_pos[1].array_get()]
        return np.array([self.obj_pos[0].array_get()[ptcl_idx], 
                         self.obj_pos[1].array_get()[ptcl_idx]])
    
    def set_particles_vel(self, T=None, ptcl_idx=None, vel_vector=None):
        if ptcl_idx is not None and vel_vector is not None:
            self.obj_vel[0].array_set(idx=ptcl_idx, value=vel_vector[0])
            self.obj_vel[1].array_set(idx=ptcl_idx, value=vel_vector[1])
            self.obj_vel[2].array_set(idx=ptcl_idx, value=vel_vector[2])
        else:
            sigma = np.sqrt(k*T/(self.mass/self.PtclperMacro))
            #self.vel = sigma * np.random.randn(self.Ntot, 3)
            self.obj_vel[0].array_set(sigma * np.random.randn(self.Ntot))
            self.obj_vel[1].array_set(sigma * np.random.randn(self.Ntot))
            self.obj_vel[2].array_set(sigma * np.random.randn(self.Ntot))
    
    def get_particles_vel(self, ptcl_idx=None):
        if self.Ntot == 0:
            print "No particles in the system"
            return
        if ptcl_idx is None:
            return [self.obj_vel[0].array_get(), self.obj_vel[1].array_get(), 
                self.obj_vel[2].array_get()]
        return np.array([self.obj_vel[0].array_get()[ptcl_idx], self.obj_vel[1].array_get()[ptcl_idx], 
                self.obj_vel[2].array_get()[ptcl_idx]])
    
    def add_particles(self, positions, velocities):
        if positions.ndim == 1:
            #positions = positions.reshape(1, -1)
            #velocities = velocities.reshape(1, -1)
            for i in range(3):
                if i < 2:
                    self.obj_pos[i].append(positions[i])
                self.obj_vel[i].append(velocities[i])
            self.Ntot += 1
        else:
            for i in range(3):
                if i < 2:
                    self.obj_pos[i].extend(positions)
                self.obj_vel[i].extend(velocities)
            self.Ntot += len(positions)
        #self.pos = np.concatenate((self.pos, positions), axis = 0)
        #self.vel = np.concatenate((self.vel, velocities), axis = 0)
    
    def Particle_pusher(self, E, B, dt, ptcl_boundary_conditions):
        '''
        pos_z = self.pos[:, 0]
        pos_r = self.pos[:, 1]
        pos_y = self.pos[:, 2]
        vel_z = self.vel[:, 0]
        vel_r = self.vel[:, 1]
        vel_y = self.vel[:, 2]
        '''
        pos_z = self.obj_pos[0].array_get()
        pos_r = self.obj_pos[1].array_get()
        vel_z = self.obj_vel[0].array_get()
        vel_r = self.obj_vel[1].array_get()
        vel_y = self.obj_vel[2].array_get()
        Ez = E[0]
        Er = E[1]
        Ey = np.zeros_like(Ez)
        Bz = B[0]
        Br = B[1]
        By = np.zeros_like(Bz)
        q = self.charge
        m = self.mass
        Ntot = self.Ntot
        ptcl_boundary_conditions_z = ptcl_boundary_conditions[0]
        ptcl_boundary_conditions_r = ptcl_boundary_conditions[1]
        motion_integrator.ParticlePush(pos_z, pos_r, vel_z, 
                        vel_r, vel_y, Ez, Er, Ey, 
                       Bz, Br, By, dt, q, m, Ntot, ptcl_boundary_conditions_z, 
                       ptcl_boundary_conditions_r, self.internal_ptcls_idx)
        
    def charge_interpolation(self):
        rho_grid = np.zeros((self.Nz, self.Nr))
        Ntot = self.Ntot
        pos_z = self.obj_pos[0].array_get()
        pos_r = self.obj_pos[1].array_get()
        rho_grid = charge_interpolation_numba(rho_grid, pos_z, pos_r, 
                                self.dz, self.dr, Ntot)
        #rho_grid = rho_grid.reshape(self.Nz, self.Nr)
        volume = self._node_volume()
        rho_grid = rho_grid / volume
        return rho_grid * self.charge
    
    def E_interpolation(self, E):
        Ez_interp = np.zeros(self.Ntot)
        Er_interp = np.zeros(self.Ntot)
        pos_z = self.obj_pos[0].array_get()
        pos_r = self.obj_pos[1].array_get()
        return E_interpolation_numba(E, Ez_interp, Er_interp, pos_z, pos_r, self.dz, self.dr, self.Ntot)
    
    def B_interpolation(self, B):
        Bz_interp = np.zeros(self.Ntot)
        Br_interp = np.zeros(self.Ntot)
        pos_z = self.obj_pos[0].array_get()
        pos_r = self.obj_pos[1].array_get()
        return B_interpolation_numba(B, Bz_interp, Br_interp, pos_z, pos_r, self.dz, self.dr, self.Ntot)
    
    
    def _R(self, j):
        return j*self.dr
    
    def _node_volume(self):
        node_vol = np.zeros([self.Nz, self.Nr])
        return method_node_volume_numba(self.dz, self.dr, node_vol)
    
if __name__ == '__main__':
    pass