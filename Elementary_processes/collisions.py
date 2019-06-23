#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:25:35 2019

@author: vladimirsmirnov
"""
import math
import numpy as np
import random

def isotropic_vector(vector_module):
    vector = np.zeros(3)
    theta = np.pi*random.uniform(-1., 1.)
    phi = np.pi*random.uniform(-1., 1.)
    vector[0] = vector_module*math.cos(theta)*math.cos(phi)
    vector[1] = vector_module*math.cos(theta)*math.sin(phi)
    vector[2] = vector_module*math.sin(theta)
    return vector
    

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
        
    def nanbu_method(self, ions):
        '''
        determine the fraction of ptcls for which the collisions will be assigned
        with the nanbu algorithm
        '''
        fraction = ions
        return fraction
    
class ElectronNeutralElasticCollision(MonteCarloCollision):
    def __init__(self, sigma, dt, gas):
        super(ElectronNeutralElasticCollision, self).__init__(sigma, dt, gas)
    
    def vel_update(self, electrons):
        assert electrons.mass/self.gas.mass < 100, "electrons.mass/self.gas.mass > 100"
        for vel in electrons.vel:
            prob = self.sigma * self.gas.n * np.linalg.norm(vel) * self.dt
            if random.random() < prob:
                print "ElectronNeutralElasticCollision occurs"
                theta = np.pi*random.uniform(-1., 1.)
                phi = np.pi*random.uniform(-1., 1.)
                vel_module = np.linalg.norm(vel)*math.sqrt(1. - 2*electrons.mass/self.gas.mass*(1 - math.cos(theta)))
                vel[0] = vel_module*math.cos(theta)*math.cos(phi)
                vel[1] = vel_module*math.cos(theta)*math.sin(phi)
                vel[2] = vel_module*math.sin(theta)
                
class IonNeutralElasticCollision(MonteCarloCollision):
    def __init__(self, sigma, dt, gas):
        super(IonNeutralElasticCollision, self).__init__(sigma, dt, gas)
        
    def vel_update(self, ions):
        for vel in ions.vel:
            prob = self.sigma*self.gas.n*np.linalg.norm(vel)*self.dt
            if random.random() < prob:
                print "IonNeutralElasticCollision occurs"
                gas_vel = self.gas.gen_vel_vector()
                R = isotropic_vector(1)
                delta_p = np.linalg.norm(vel - gas_vel)*self.gas.mass*R
                vel = (delta_p + ions.mass*vel + self.gas.mass*gas_vel) / (ions.mass + self.gas.mass)

class Ionization(MonteCarloCollision):
    def __init__(self, sigma, dt, gas, I):
        super(Ionization, self).__init__(sigma, dt, gas)
        self.ion_threshold = I
    
    def vel_update(self, electrons, ions):
        #new_electron_pos = []
        #new_electron_vel = []
        for idx, vel in enumerate(electrons.vel):
            prob = self.sigma*self.gas.n*np.linalg.norm(vel)*self.dt
            if random.random() < prob:
                print "Ionization occurs"
                gas_vel = self.gas.gen_vel_vector()
                mu = self.gas.mass*electrons.mass/(self.gas.mass + electrons.mass)
                deltaE = 0.5*mu*(np.linalg.norm(vel - gas_vel))**(2) - self.ion_threshold
                if deltaE < 0:
                    continue
                prop = random.uniform(0, 1)
                vel1_module = math.sqrt(2*prop*deltaE/electrons.mass)
                vel2_module = math.sqrt(2*(1-prop)*deltaE/electrons.mass)
                electrons.vel[idx] = isotropic_vector(vel1_module)
                #vel = isotropic_vector(vel1_module)
                electrons.add_particles(electrons.pos[idx], isotropic_vector(vel2_module))
                ions.add_particles(electrons.pos[idx], gas_vel)
                #new_electron_vel.append(isotropic_vector(vel2_module))
                #new_electron_pos.append(electrons.pos[idx])
        #if new_electron_pos:
        #    electrons.add_particles(np.array(new_electron_pos), np.array(new_electron_vel))
            #ions.add_particles()