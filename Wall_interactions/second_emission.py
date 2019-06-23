#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:53:31 2019

@author: vladimirsmirnov
"""
import random
import numpy as np
import math
import numba

#@numba.njit(cache=True)
def select_particles(massive1, ptcl_idx):
    return massive1[ptcl_idx]

@numba.njit(cache=True)
def incident_particles(pos_z, pos_r, boundary_conditions_z, boundary_conditions_r):
    mask1 = pos_z < boundary_conditions_z[0] or pos_z > boundary_conditions_z[1]
    mask2 = pos_r > boundary_conditions_r[0] and pos_r < boundary_conditions_r[1]
    if mask1 == True and mask2 == True:
        return True
    return False

@numba.njit(cache=True)
def determine_direction(pos_z, boundary_conditions_z):
    if pos_z < boundary_conditions_z[0]:
        return 1
    if pos_z > boundary_conditions_z[1]:
        return -1

@numba.njit(cache=True, fastmath=True)      
def filter_particles(pos_z, pos_r, Ntot, boundary_conditions_z, boundary_conditions_r):
    ptcl_idx = []
    for ip in range(Ntot):
        if incident_particles(pos_z[ip], pos_r[ip], boundary_conditions_z,
                                  boundary_conditions_r) == True:
            ptcl_idx.append(ip)
    return ptcl_idx

class SecondEmission(object):
    def __init__(self, incident_ptcls, emitted_ptcls, boundary_conditions, gamma, emission_energy):
        self.incident_ptcls = incident_ptcls
        self.emitted_ptcls = emitted_ptcls
        self.boundary_conditions = boundary_conditions
        self.gamma = gamma
        self.emission_energy = emission_energy
        
    def particle_emission(self):
        #print "Ntot:", self.incident_ptcls.Ntot
        incident_ptcl_idx = filter_particles(self.incident_ptcls.obj_pos[0].array_get(), 
                                self.incident_ptcls.obj_pos[1].array_get(),
                               self.incident_ptcls.Ntot, self.boundary_conditions[0], 
                               self.boundary_conditions[1])
        #print incident_ptcl_idx
        incident_pos_z = select_particles(self.incident_ptcls.obj_pos[0].array_get(), incident_ptcl_idx)
        incident_pos_r = select_particles(self.incident_ptcls.obj_pos[1].array_get(), incident_ptcl_idx)
        for idx, pos_z in enumerate(incident_pos_z):
            if random.uniform(0, 1) < self.gamma:
                vel_module = np.sqrt(2.*self.emission_energy/(self.emitted_ptcls.mass/self.emitted_ptcls.PtclperMacro))
                direction = determine_direction(pos_z, self.boundary_conditions[0])
                vel_vector = direction*np.array([vel_module, 0, 0])
                if direction == 1:
                    pos = np.array([self.boundary_conditions[0][0], incident_pos_r[idx]])
                else:
                    pos = np.array([self.boundary_conditions[0][1], incident_pos_r[idx]])
                self.emitted_ptcls.add_particles(pos, vel_vector)
        '''
        incident_pos = select_particles(self.incident_ptcls.pos, incident_ptcl_idx)
        for pos in incident_pos:
            if random.uniform(0, 1) < self.gamma:
                vel_module = math.sqrt(2*self.emission_energy/self.emitted_ptcls.mass)
                direction = determine_direction(pos[0], self.boundary_conditions[0])
                vel_vector = direction*np.array([vel_module, 0, 0])
                self.emitted_ptcls.add_particles(pos, vel_vector)
        '''
    '''  
    def particle_leave_emission(self, emmited_ptcls, cathodeR):
        ptcl_pos_vel = np.concatenate((self.ptcls.pos, self.ptcls.vel), axis = 1)
        # part of particles near cathode surface
        ptcl_pos_vel_surface = ptcl_pos_vel[((ptcl_pos_vel[:,0] < self.leave_conditions[0]) | 
                            (ptcl_pos_vel[:,0] > self.leave_conditions[1]))]
        # part of particles inside the volume
        ptcl_pos_vel_volume = ptcl_pos_vel[(ptcl_pos_vel[:,0] > self.leave_conditions[0]) & 
                            (ptcl_pos_vel[:,0] < self.leave_conditions[1])]
        self.ptcls.pos = ptcl_pos_vel_volume[:,:3]
        self.ptcls.vel = ptcl_pos_vel_volume[:,3:]
        self.ptcls.Ntot = ptcl_pos_vel_volume.shape[0]
        new_vel_list = []
        new_ptcl_idx = []
        for i in range(ptcl_pos_vel_surface.shape[0]):
            if random.uniform(0, 1) < self.gamma and ptcl_pos_vel_surface[i][1] > cathodeR:
                #print "emission occurs"
                vel_module = math.sqrt(2*self.energy_distr/emmited_ptcls.mass)
                #new_vel = self.isotropic_vector(vel_module)
                
                if abs(self.leave_conditions[0] - ptcl_pos_vel_surface[i,0]) < abs(self.leave_conditions[1] - ptcl_pos_vel_surface[i,0]):
                    sign = 1
                else:
                    sign = -1
                
                new_vel = sign*np.array([vel_module, 0., 0.])
                new_vel_list.append(new_vel)
                new_ptcl_idx.append(i)
        # add new emmited particles to others
        if new_vel_list:
            emmited_ptcls.add_particles(positions = ptcl_pos_vel_surface[new_ptcl_idx,:3], 
                          velocities = np.array(new_vel_list))
        return self.ptcls, emmited_ptcls
    '''       
    @staticmethod
    def isotropic_vector(vector_module):
        vector = np.zeros(3)
        theta = np.pi*random.uniform(-1., 1.)
        phi = np.pi*random.uniform(-1., 1.)
        vector[0] = vector_module*math.cos(theta)*math.cos(phi)
        vector[1] = vector_module*math.cos(theta)*math.sin(phi)
        vector[2] = vector_module*math.sin(theta)
        return vector
    