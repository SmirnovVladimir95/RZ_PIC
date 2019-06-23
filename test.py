#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file is a test of full-scale multiparticle 
        calculation of Penning discharge process
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

def main():
    Nz = 200
    Nr = 70
    dz = 2e-5
    dr = 2e-5
    Ar_mass = 6.6335209e-26
    #Ar_mass = 100*m_e
    B_const = 10 # 1 Tesla = 10000 Gauss
    field_tolerance = 1e-3
    Ar_plus_step = 100 # iter step for Ar+
    
    electron = Particle(mass = m_e, charge = -1*e, dens = 1e12, 
                PtclperCell = 100, Nz = Nz, Nr = Nr, dz = dz, 
                 dr = dr)
    Ar_plus = Particle(mass = 100*m_e, charge = e, dens = 1e12, 
                PtclperCell = 100, Nz = Nz, Nr = Nr, dz = dz, 
                dr = dr)
    Ar = NeutralGas(T=500., n=1e20, mass=Ar_mass)
    
    fld = Field(cathodePhi = -100., anodePhi = 0., cathodeR = int(0.5*Nr),
                             Nz = Nz, Nr = Nr, dz = dz, 
                             dr = dr, B_const=B_const)
    
    Ar_plus.set_particles_pos(seed=10)
    electron.set_particles_pos(seed=20)
    Ar_plus.set_particles_vel(T=500.)
    electron.set_particles_vel(T=e/k)
    
    
    rho_i = Ar_plus.charge_interpolation()
    rho_e = electron.charge_interpolation()
    # sns.heatmap(rho_e.T)
    
    phi = fld.Poisson_solver(rho_e=rho_e, rho_i=rho_i, tolerance=field_tolerance)
    leave_conditions = np.zeros((2, 2))
    leave_conditions[0, 0] = 1*dz
    leave_conditions[0, 1] = Nz*dz - 2*dz
    leave_conditions[1, 0] = -1
    leave_conditions[1, 1] = Nr*dr - dr
    
    emission_conditions = np.zeros((2, 2))
    emission_conditions[0, 0] = 1*dz
    emission_conditions[0, 1] = Nz*dz - 2*dz
    emission_conditions[1, 0] = 0.
    emission_conditions[1, 1] = 0.5*Nr*dr
    
    Ar_plus_leave = SimpleLeave(ptcls=Ar_plus, leave_conditions=leave_conditions)
    electron_leave = SimpleLeave(ptcls=electron, leave_conditions=leave_conditions)
    electron_emission = SecondEmission(Ar_plus, electron, emission_conditions, gamma=0.1, emission_energy=10.*e)
    
    max_iter = 100
    dt = 1e-12
    
    Ar_plus_elastic_collisions = IonNeutralElasticCollision(sigma=1e-20, dt=dt*Ar_plus_step, gas=Ar, ions=Ar_plus)
    electron_elastic_collisions = ElectronNeutralElasticCollision(sigma=1e-19, dt=dt, gas=Ar, electrons=electron)
    electron_Ar_ionization = Ionization(sigma=1e-20, dt=dt, gas=Ar, I=10*e, electrons=electron, ions=Ar_plus)
    v_Ar_plus_max = np.linalg.norm(np.array([Ar_plus.get_particles_vel()[0].max(), 
                        Ar_plus.get_particles_vel()[1].max(),
                        Ar_plus.get_particles_vel()[2].max()]))
    v_electron_max = np.linalg.norm(np.array([electron.get_particles_vel()[0].max(), 
                        electron.get_particles_vel()[1].max(),
                        electron.get_particles_vel()[2].max()]))
    Ar_plus_collisions = NullCollisions(v_max=v_Ar_plus_max, collisions=[Ar_plus_elastic_collisions])
    electron_collisions = NullCollisions(v_max=v_electron_max, collisions = [electron_elastic_collisions, electron_Ar_ionization])
    
    
    E = fld.compute_EF(phi=phi)
        
    Bz_interp_electron, Br_interp_electron = electron.B_interpolation(fld.B)
    Ez_interp_electron, Er_interp_electron = electron.E_interpolation(E)
    Bz_interp_Ar_plus, Br_interp_Ar_plus = Ar_plus.B_interpolation(fld.B)
    Ez_interp_Ar_plus, Er_interp_Ar_plus = Ar_plus.E_interpolation(E)
    electron.Particle_pusher(E=(Ez_interp_electron, Er_interp_electron), 
                B=(Bz_interp_electron, Br_interp_electron), dt=-0.5*dt, 
                ptcl_boundary_conditions=leave_conditions)
    Ar_plus.Particle_pusher(E=(Ez_interp_Ar_plus, Er_interp_Ar_plus), 
                B=(Bz_interp_Ar_plus, Br_interp_Ar_plus), dt=-0.5*dt*Ar_plus_step, 
                ptcl_boundary_conditions=leave_conditions)
    
    electron_emission.particle_emission()
    
    Ar_plus_leave.particle_leave()
    electron_leave.particle_leave()
    
    #Ar_plus, electron = electron_2_emmision.particle_leave_emission(electron, cathodeR=0.5*Nr*dr)
    #Ar_plus_leave.particle_leave()
    #electron_leave.particle_leave()
    #phi = None
    open("test.txt","w").close()
    fout = open("test.txt","a")
    Ar_plus_Ntot = np.zeros(max_iter)
    electron_Ntot = np.zeros(max_iter)
    electron_track_pos = np.zeros((max_iter, 2))
    t0 = time.time()
    for it in range(max_iter):
        #fout.write(str(electron.Ntot) + '\n')
        if it % 500 == 0:
            print it
        #print Ar_plus.Ntot, electron.Ntot
        Ar_plus_Ntot[it] = Ar_plus.Ntot
        electron_Ntot[it] = electron.Ntot
        electron_track_pos[it,:] = electron.get_particles_pos(ptcl_idx=int(1e5))
        electron_emission.particle_emission()
        if it % Ar_plus_step == 0:
            Ar_plus_leave.particle_leave()
        electron_leave.particle_leave()
        
        if it % Ar_plus_step == 0:
            Ar_plus_collisions.collision()
        electron_collisions.collision()
        
        rho_e = electron.charge_interpolation()
        if it % Ar_plus_step == 0:
            rho_i = Ar_plus.charge_interpolation()
        
        phi = fld.Poisson_solver(phi=phi, rho_e=rho_e, rho_i=rho_i, 
                                 tolerance=field_tolerance)
        E = fld.compute_EF(phi=phi)
        Bz_interp_electron, Br_interp_electron = electron.B_interpolation(fld.B)
        Ez_interp_electron, Er_interp_electron = electron.E_interpolation(E)
        if it % Ar_plus_step == 0:
            Bz_interp_Ar_plus, Br_interp_Ar_plus = Ar_plus.B_interpolation(fld.B)
            Ez_interp_Ar_plus, Er_interp_Ar_plus = Ar_plus.E_interpolation(E)
        
        
        electron.Particle_pusher(E=(Ez_interp_electron, Er_interp_electron), 
                B=(Bz_interp_electron, Br_interp_electron), dt=dt, 
                ptcl_boundary_conditions=leave_conditions)
        if it % Ar_plus_step == 0:
            Ar_plus.Particle_pusher(E=(Ez_interp_Ar_plus, Er_interp_Ar_plus), 
                B=(Bz_interp_Ar_plus, Br_interp_Ar_plus), dt=dt*Ar_plus_step, 
                ptcl_boundary_conditions=leave_conditions)
    
    fout.write(str(time.time() - t0))
    fout.close()
    t1 = time.time()
    print "time per iter:", (t1 - t0)/max_iter  
    #plt.plot(range(Ar_plus_Ntot.shape[0]), Ar_plus_Ntot)
    #plt.plot(range(electron_Ntot.shape[0]), electron_Ntot)
    #plt.scatter(electron_track_pos[:,0], electron_track_pos[:,1])
    plt.plot(electron_track_pos[:,0], electron_track_pos[:,1])
    #plt.xlim(0, Nz*dz)
    #plt.ylim(0, Nr*dr)
    #plt.xlabel("iter")
    #plt.ylabel("N")
    #plt.savefig("N(iter).png")
    
if __name__ == '__main__':
    main()
    
    
    
    
    
