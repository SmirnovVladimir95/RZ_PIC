#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:28:13 2019

@author: vladimirsmirnov
"""
import numpy as np

import scipy.constants as const

import seaborn as sns

import time

from numba_func import gauss_seidel_integrator

from tensor_interpolation import tensor_interpolation

import matplotlib.pyplot as plt

from Poisson_solver_Grad_method import Poisson_solver_GRD_METHOD

class Field():
    def __init__(self, cathodePhi, anodePhi, cathodeR, Nz, Nr, dz, dr, B_const=None):
        self.Nz = Nz
        self.Nr = Nr
        self.dz = dz
        self.dr = dr
        self.cathodePhi = cathodePhi
        self.anodePhi = anodePhi
        self.cathodeR = cathodeR
        self.B = self.set_magnetic_Field(B_const) if type(B_const) == int or float else B_const
        # internal parameters:
        self._phi_dirichlet = None
        self._cell_type = None
        self._r = None
    
    def _set_init_phi_cell_type(self, phi):
        nz = self.Nz
        nr = self.Nr
        #self._phi_dirichlet = np.zeros([self.Nz, self.Nr], float)
        self._cell_type = np.zeros([nz,nr])
        for iz in range(nz):
            for ir in range(nr):
                if iz == nz - 1 and ir >= 0 and ir < self.cathodeR:
                    phi[iz][ir] = self.cathodePhi
                    self._cell_type[iz][ir] = 1
                if iz == 0 and ir >= 0 and ir < self.cathodeR:
                    phi[iz][ir] = self.cathodePhi
                    self._cell_type[iz][ir] = 1
                if ir == nr - 1:
                    phi[iz][ir] = self.anodePhi
                    self._cell_type[iz][ir] = 1
        return phi, self._cell_type
    
    def _set_radia(self, phi):
        nz = self.Nz
        nr = self.Nr
        R = lambda j: j*self.dr
        r = np.zeros_like(phi)
        for i in range(nz):
            for j in range(nr):
                r[i][j] = R(j)
        return r
    
    def _rho_vector(self, rho_e = 0., rho_i = 0.):
        return np.where(self._cell_type<=0,(rho_i + rho_e)/const.epsilon_0, 0.)
    
    def Poisson_solver(self, phi = None, rho_e = 0., rho_i = 0., tolerance = 1e-5, max_iter = 1000000):
        # set initial value for Phi and constant value for _cell_type 
        if phi is None:
            phi = np.zeros([self.Nz, self.Nr], float)
            phi, self._cell_type = self._set_init_phi_cell_type(phi)
        if self._cell_type is None:
            phi, self._cell_type = self._set_init_phi_cell_type(phi)
        #make copy of dirichlet nodes
        P = np.copy(phi)
        g = np.zeros_like(phi)
        #phi = np.copy(init_approx)
        dz2 = self.dz*self.dz
        dr2 = self.dr*self.dr
        #set radia
        if self._r is None:
            self._r = self._set_radia(phi)
        b = self._rho_vector(rho_e = rho_e, rho_i = rho_i)
        g = np.where(self._cell_type>0, P, 0)
        for it in xrange(max_iter):
            
            g = gauss_seidel_integrator(g, phi, b, self._r, self.dr, dr2, self.dz, dz2)
            
            if it % 50 == 0:
                phi_new = np.where(self._cell_type>0,P,g)
                current_coef = self._convergence(phi_new, phi, tolerance)
                if current_coef == True:
                    #print "current_coef:", convergence_level
                    #print "tolerance level achieved!"
                    #print "iteration number: ", it
                    #print T/it
                    return phi_new

            # neumann boundaries
            g[:,0] = g[:,1] # on the axes (r = 0)
            g[0,self.cathodeR:] = g[1,self.cathodeR:] # on the one side after cathode
            g[g.shape[0] - 1, self.cathodeR:] = g[g.shape[0] - 2, self.cathodeR:] # on the other side after cathode
    
            # dirichlet boundaries
            phi = np.where(self._cell_type>0,P,g)
        return phi
    
    def Poisson_solver_GRD_METHOD(self, phi=None, rho_e = 0., rho_i = 0., tolerance = 1e-5):
        # set initial value for Phi and constant value for _cell_type 
        if phi is None:
            phi = np.zeros([self.Nz, self.Nr], float)
        if self._cell_type is None:
            phi, self._cell_type = self._set_init_phi_cell_type(phi)
        b = self._rho_vector(rho_e = rho_e, rho_i = rho_i)
        return Poisson_solver_GRD_METHOD(phi_init=phi, b=b, CathodeV=self.cathodePhi, AnodeV=self.anodePhi, Nz=self.Nz, 
                                         Nr=self.Nr, dz=self.dz, dr=self.dr, CathodeR=self.cathodeR, 
                                         tol=tolerance)
    
    def _convergence(self, X1, X2, tol):
        deltaX = X1 - X2
        if deltaX[abs(deltaX) > tol].shape[0] != 0:
            return False
        return True
    
    def compute_EF(self, phi):
        nz = self.Nz
        nr = self.Nr
        dz = self.dz
        dr = self.dr
        efz = np.zeros([nz,nr])
        efr = np.zeros([nz,nr])
        #central difference, not right on walls
        efz[1:-1] = (phi[0:nz-2]-phi[2:nz+1])/(2*dz)
        efr[:,1:-1] = (phi[:,0:nr-2]-phi[:,2:nr+1])/(2*dr)
        #one sided difference on boundaries
        efz[0,:] = (phi[0,:]-phi[1,:])/dz
        efz[-1,:] = (phi[-2,:]-phi[-1,:])/dz
        efr[:,0] = (phi[:,0]-phi[:,1])/dr
        efr[:,-1] = (phi[:,-2]-phi[:,-1])/dr
        return efz, efr
    
    def set_magnetic_Field(self, B_const):
        Bz = np.empty([self.Nz, self.Nr])
        Bz.fill(B_const)
        Br = np.zeros([self.Nz, self.Nr])
        return np.array([Bz, Br])
    
if __name__ == '__main__':
    print "Simple Iteration"
    multy_coeff = 1
    field_obj = Field(cathodePhi=-100., anodePhi=0., cathodeR=25*multy_coeff, 
                      Nz=100*multy_coeff, 
                      Nr=50*multy_coeff, dz=2e-5, dr=2e-5, B_const=0.1)
    t0 = time.time()
    phi_init = np.ones((100*multy_coeff, 50*multy_coeff))*(-50)
    phi = field_obj.Poisson_solver(phi=phi_init, tolerance=1e-3)
    phi_1 = field_obj.Poisson_solver(phi=phi_init, tolerance=1e-1)
    #phi = field_obj.Poisson_solver_GRD_METHOD(phi=phi, tolerance=1e-5)
    t1 = time.time()
    print t1 - t0
    sns.heatmap((phi-phi_1).T)
    '''
    print "MultyGrid"
    
    t0 = time.time()
    phi_init = np.ones((100*multy_coeff, 50*multy_coeff))*(-50)
    coarse = 4
    phi_coarsed, grid_coarsed = tensor_interpolation(tensor=phi_init, coeff=coarse, Type='coarse', method='cubic')

    fld_coarsed = Field(cathodePhi=-100, anodePhi=0, cathodeR=int(phi_coarsed.shape[1]*0.5),
                  Nz=phi_coarsed.shape[0], Nr=phi_coarsed.shape[1], dz=2e-5*coarse, dr=2e-5*coarse, B_const=0.1)
    phi_coarsed = fld_coarsed.Poisson_solver(phi=phi_coarsed)
    
    phi_fine, grid_fine = tensor_interpolation(tensor=phi_coarsed, 
                            coeff=coarse, Type='fine', grid=grid_coarsed, method='cubic')
    fld_fine = Field(cathodePhi=-100, anodePhi=0, 
                  cathodeR=int(phi_fine.shape[1]*0.5),
                  Nz=phi_fine.shape[0], Nr=phi_fine.shape[1], dz=2e-5, 
                  dr=2e-5, B_const=0.1)
    phi_fine = fld_fine.Poisson_solver(phi=phi_fine)
    t1 = time.time()
    print t1-t0
    import seaborn as sns
    sns.heatmap((phi_fine - phi).T)
    '''