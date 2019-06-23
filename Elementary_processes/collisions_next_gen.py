#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 00:32:44 2019

@author: vladimirsmirnov
"""
import numpy as np

class MonteCarloCollision(object):
    def __init__(self, sigma, dt, gas):
        '''
        sigma - reaction cross section
        n - the concentration of neutral gas
        dt - the integration step at which the collision probability is estimated
        mass - mass of neutral gas
        '''
        self.sigma = sigma
        self.gas = gas
        self.dt = dt
        
class ElectronNeutralElasticCollision(MonteCarloCollision):
    
    """Elastic collision"""
    
    def __init__(self, sigma, dt, gas, electrons):
        super(ElectronNeutralElasticCollision, self).__init__(sigma, dt, gas)
        self.ptcls = electrons
    
    def vel_update(self, ptcl_idx):
        #assert self.gas.mass/self.ptcls.mass > 100, "electrons.mass/self.gas.mass > 100"
        #print "ElectronNeutralElasticCollision occurs"
        theta = np.pi*np.random.uniform(-1., 1.)
        phi = np.pi*np.random.uniform(-1., 1.)
        vel_vector = self.ptcls.get_particles_vel(ptcl_idx)
        vel_module = np.linalg.norm(vel_vector)*np.sqrt(1. - 2*self.ptcls.mass/self.gas.mass*(1 - np.cos(theta)))
        vel_vector[0] = vel_module*np.cos(theta)*np.cos(phi)
        vel_vector[1] = vel_module*np.cos(theta)*np.sin(phi)
        vel_vector[2] = vel_module*np.sin(theta)
        self.ptcls.set_particles_vel(ptcl_idx=ptcl_idx, vel_vector=vel_vector)
    
class IonNeutralElasticCollision(MonteCarloCollision):
    
    """Elastic collision"""
    
    def __init__(self, sigma, dt, gas, ions):
        super(IonNeutralElasticCollision, self).__init__(sigma, dt, gas)
        self.ptcls = ions
        
    def vel_update(self, ptcl_idx):
        #print "IonNeutralElasticCollision occurs"
        gas_vel = self.gas.gen_vel_vector()
        R = isotropic_vector(1)
        ion_vel = self.ptcls.get_particles_vel(ptcl_idx)
        delta_p = np.linalg.norm(ion_vel - gas_vel)*self.gas.mass*R
        new_ion_vel = (delta_p + self.ptcls.mass*ion_vel + self.gas.mass*gas_vel) / (self.ptcls.mass + self.gas.mass)
        self.ptcls.set_particles_vel(ptcl_idx=ptcl_idx, vel_vector=new_ion_vel)

class Ionization(MonteCarloCollision):
    
    """Ionization"""
    
    def __init__(self, sigma, dt, gas, I, electrons, ions):
        super(Ionization, self).__init__(sigma, dt, gas)
        self.ion_threshold = I
        self.ptcls = electrons
        self.ions = ions
    
    def vel_update(self, ptcl_idx):
        #print "Ionization occurs"
        gas_vel = self.gas.gen_vel_vector()
        mu = self.gas.mass*self.ptcls.mass/(self.gas.mass + self.ptcls.mass)
        electron_vel = self.ptcls.get_particles_vel(ptcl_idx)
        deltaE = 0.5*mu*(np.linalg.norm(electron_vel - gas_vel))**(2) - self.ion_threshold
        if deltaE < 0:
            return
        prop = np.random.uniform(0, 1)
        vel1_module = np.sqrt(2*prop*deltaE/self.ptcls.mass)
        vel2_module = np.sqrt(2*(1-prop)*deltaE/self.ptcls.mass)
        pos = self.ptcls.get_particles_pos(ptcl_idx)
        self.ptcls.set_particles_vel(ptcl_idx=ptcl_idx, vel_vector=isotropic_vector(vel1_module))
        self.ptcls.add_particles(pos, isotropic_vector(vel2_module))
        self.ions.add_particles(pos, gas_vel)
        

def isotropic_vector(vector_module):
    vector = np.zeros(3)
    theta = np.pi*np.random.uniform(-1., 1.)
    phi = np.pi*np.random.uniform(-1., 1.)
    vector[0] = vector_module*np.cos(theta)*np.cos(phi)
    vector[1] = vector_module*np.cos(theta)*np.sin(phi)
    vector[2] = vector_module*np.sin(theta)
    return vector

