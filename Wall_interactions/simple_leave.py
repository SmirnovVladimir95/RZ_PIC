#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:04:55 2019

@author: vladimirsmirnov
"""
import numpy as np
import time
import numba

@numba.njit(cache=True)
def select_particles(massive1, ptcl_idx):
    return massive1[ptcl_idx]
'''
@numba.njit(cache=True)
def _select_particles(massive1, massive2, ptcl_idx, new_massive1, new_massive2):
    for idx, item in enumerate(ptcl_idx):
        new_massive1[idx] = massive1[item]
        new_massive2[idx] = massive2[item]
    return new_massive1, new_massive2
'''

@numba.njit(cache=True)
def internal_particles(pos_z, pos_r, boundary_conditions_z, boundary_conditions_r):
    mask1 = pos_z >= boundary_conditions_z[0] and pos_z <= boundary_conditions_z[1]
    mask2 = pos_r >= boundary_conditions_r[0] and pos_r <= boundary_conditions_r[1]
    if mask1 == True and mask2 == True:
        return True
    return False

@numba.njit(cache=True)      
def filter_particles(pos_z, pos_r, Ntot, boundary_conditions_z, boundary_conditions_r, ptcl_idx):
    #ptcl_idx = []
    for ip in range(Ntot):
        if internal_particles(pos_z[ip], pos_r[ip], boundary_conditions_z,
                                  boundary_conditions_r) == True:
            #ptcl_idx.append(ip)
            ptcl_idx[ip] = True
        else:
            ptcl_idx[ip] = False
    return ptcl_idx

class SimpleLeave(object):
    def __init__(self, ptcls, leave_conditions):
        self.ptcls = ptcls
        self.leave_conditions = leave_conditions
    
    def particle_leave(self):
        '''
        self.ptcls.pos, self.ptcls.vel = select_particles(self.ptcls.pos, 
                            self.ptcls.vel, self.ptcls.internal_ptcls_idx)
        '''
        #print "Ntot leave:", self.ptcls.Ntot
        internal_ptcls_idx = np.empty(self.ptcls.Ntot, dtype=bool)
        internal_ptcls_idx = filter_particles(self.ptcls.obj_pos[0].array_get(), 
                                self.ptcls.obj_pos[1].array_get(),
                               self.ptcls.Ntot, self.leave_conditions[0], 
                               self.leave_conditions[1], internal_ptcls_idx)
        
        self.ptcls.obj_pos[0].array_set(select_particles(self.ptcls.obj_pos[0].array_get(), internal_ptcls_idx))
        self.ptcls.obj_pos[1].array_set(select_particles(self.ptcls.obj_pos[1].array_get(), internal_ptcls_idx))
        
        self.ptcls.obj_vel[0].array_set(select_particles(self.ptcls.obj_vel[0].array_get(), internal_ptcls_idx))
        self.ptcls.obj_vel[1].array_set(select_particles(self.ptcls.obj_vel[1].array_get(), internal_ptcls_idx))
        self.ptcls.obj_vel[2].array_set(select_particles(self.ptcls.obj_vel[2].array_get(), internal_ptcls_idx))
        self.ptcls.Ntot = self.ptcls.obj_pos[0].array_size()
        #self.ptcls.internal_ptcls_idx = np.ones(self.ptcls.Ntot, dtype=bool)
        #self.ptcls.internal_ptcls_idx = []
        '''
        massive = np.concatenate((self.ptcls.pos, self.ptcls.vel), axis = 1)
        massive = massive[(massive[:,0] > self.leave_conditions[0][0]) & (massive[:,0] < self.leave_conditions[0][1]) &
                    (massive[:,1] > self.leave_conditions[1][0]) & (massive[:,1] < self.leave_conditions[1][1])]
        self.ptcls.pos = massive[:,0:3]
        self.ptcls.vel = massive[:,3:]
        self.ptcls.Ntot = massive.shape[0]
        '''
    
if __name__ == '__main__':
    N = int(1e6)
    dim = 3
    
    a = np.empty((N, dim))
    b = np.empty((N, dim))
    idx = list(np.arange(int(1e5)))
    idx = np.arange(int(1e5))
    #idx = np.ones(shape=int(1e7), dtype=bool)
    #print idx
    '''
    t0 = time.time()
    a_new = np.empty((idx.shape[0], dim))
    b_new = np.empty((idx.shape[0], dim))
    select_particles(a, b, idx, a_new, b_new)
    print time.time() - t0
    '''
    t0 = time.time()
    #idx = np.array(idx)
    #idx_numpy = np.empty(shape=(len(idx)), dtype=int)
    #idx_numpy = idx
    select_particles(a, b, idx)
    print time.time() - t0
    
    
    