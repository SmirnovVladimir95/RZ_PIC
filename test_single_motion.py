#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
test of single particle motion in electromagnetic field
"""

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt 

import time

from scipy.constants import k, m_e, e 

from Particles.particle import Particle

from Particles.neutral_gas import NeutralGas

from Field.field import Field

from Elementary_processes.collision_choice import NullCollisions

from Elementary_processes.collisions_next_gen import ElectronNeutralElasticCollision, IonNeutralElasticCollision, Ionization

from Wall_interactions.second_emission import SecondEmission

from Wall_interactions.simple_leave import SimpleLeave

if __name__ == '__main__':
    Nz = 100
    Nr = 100
    dz = 2e-5
    dr = 2e-5
    B_const = 0.1
    field_tolerance = 1e-3
    electron = Particle(mass = m_e, charge = -1*e, dens = 1e16, 
                single=True, Nz = Nz, Nr = Nr, dz = dz, 
                 dr = dr)
    fld = Field(cathodePhi = -100., anodePhi = 0., cathodeR = int(0.5*Nr),
                             Nz = Nz, Nr = Nr, dz = dz, 
                             dr = dr, B_const=B_const)
    electron.set_particles_pos(z=2e-4, r=2e-4)
    print electron.get_particles_pos(ptcl_idx=0)
    E = e
    sigma = np.sqrt(3*E/(m_e))
    print sigma
    electron.set_particles_vel(ptcl_idx=0, vel_vector=np.array([sigma, 0, 0]))
    
    leave_conditions = np.zeros((2, 2))
    leave_conditions[0, 0] = 1*dz
    leave_conditions[0, 1] = Nz*dz - 2*dz
    leave_conditions[1, 0] = -1
    leave_conditions[1, 1] = Nr*dr - dr
    
    electron_leave = SimpleLeave(ptcls=electron, leave_conditions=leave_conditions)
    
    max_iter = 500
    dt = 1e-11
    
    phi = fld.Poisson_solver(phi=None, tolerance=field_tolerance)
    phi_1 = fld.Poisson_solver(phi=None, tolerance=1e-4)
    #sns.heatmap((phi-phi_1).T)
    E = fld.compute_EF(phi=phi)
    Bz_interp_electron, Br_interp_electron = electron.B_interpolation(fld.B)
    Ez_interp_electron, Er_interp_electron = electron.E_interpolation(E)
    
    electron.Particle_pusher(E=(Ez_interp_electron, Er_interp_electron), 
                B=(Bz_interp_electron, Br_interp_electron), dt=-0.5*dt, 
                ptcl_boundary_conditions=leave_conditions)
    
    z = []
    r = []
    for it in range(max_iter):
        electron_leave.particle_leave()
        if electron.Ntot == 0:
            break
        z.append(electron.get_particles_pos(ptcl_idx=0)[0])
        r.append(electron.get_particles_pos(ptcl_idx=0)[1])
        #print electron.get_particles_pos(ptcl_idx=0)
        Bz_interp_electron, Br_interp_electron = electron.B_interpolation(fld.B)
        #print Bz_interp_electron
        Ez_interp_electron, Er_interp_electron = electron.E_interpolation(E)
        #print Ez_interp_electron.shape
        #Ez_interp_electron = np.zeros_like(Ez_interp_electron)
        #Er_interp_electron = np.zeros_like(Er_interp_electron)
        #print Ez_interp_electron
        electron.Particle_pusher(E=(Ez_interp_electron, Er_interp_electron), 
                B=(Bz_interp_electron, Br_interp_electron), dt=dt, 
                ptcl_boundary_conditions=leave_conditions)
    
    #print z
    plt.plot(z, r)
    plt.scatter(z, r)
    print max(r) - min(r)
        
        
        
    
