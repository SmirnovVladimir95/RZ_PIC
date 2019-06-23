#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 08:34:21 2019

@author: vladimirsmirnov
"""
import numpy as np

class NullCollisions(object):
    def __init__(self, v_max=None, max_prob=None, collisions=[]):   
        self.collision_list = []
        if v_max is None and max_prob is None:
            print "Enter v_max or max_prob:"
            return
        if max_prob is None:
            max_prob = 0
            for collision in collisions:
                self.collision_list.append(collision)
                max_prob += collision.sigma * collision.gas.n * v_max * collision.dt
        self.max_prob = max_prob
        
    def get_collision_prob(self, ptcl_vel, coll_type_idx):
        sigma = self.collision_list[coll_type_idx].sigma
        n = self.collision_list[coll_type_idx].gas.n
        vel = np.linalg.norm(ptcl_vel)
        dt = self.collision_list[coll_type_idx].dt
        return sigma*n*vel*dt
        
    def nanbu_method(self, ptcl_vel):
        U = np.random.uniform(0, 1)
        i = int(np.floor(U*len(self.collision_list)))
        if U > float(i)/len(self.collision_list) - self.get_collision_prob(ptcl_vel, i):
            return i
        return None
    
    def simple(self):
        pass
    
    def collision(self):
        '''
        ptcls[0] - incident particles
        ptcls[1], ptcls[2]... - born particles
        '''
        #ptcl_num = ptcls[0].Ntot
        ptcl_num = self.collision_list[0].ptcls.Ntot
        ptcl_frac_size = int(ptcl_num*self.max_prob)
        ptcl_frac_idx = np.random.randint(low=0, high=ptcl_num, size=ptcl_frac_size)
        #ptcl_frac_idx = random.sample(xrange(ptcl_num), size=ptcl_frac_size)
        for ptcl_idx in ptcl_frac_idx:
            ptcl_vel = self.collision_list[0].ptcls.get_particles_vel(ptcl_idx)
            coll_idx = self.nanbu_method(ptcl_vel)
            if coll_idx is not None:
                if self.collision_list[coll_idx].__doc__ == "Elastic collision":
                    self.collision_list[coll_idx].vel_update(ptcl_idx)
                if self.collision_list[coll_idx].__doc__ == "Excitation":
                    pass
                if self.collision_list[coll_idx].__doc__ == "Ionization":
                    self.collision_list[coll_idx].vel_update(ptcl_idx)
                    
        
if __name__ == '__main__':
    pass
    