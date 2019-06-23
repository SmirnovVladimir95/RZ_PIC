#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:28:51 2019

@author: vladimirsmirnov
"""

import numpy as np
import numba
from numba import prange
import math
import time
from scipy.constants import m_e, e
#import cython

@numba.njit(cache=True, fastmath=True)
def internal_particles(ptcl_pos_z, ptcl_pos_r, boundary_conditions_z, boundary_conditions_r):
    #print boundary_conditions.shape
    z_left = boundary_conditions_z[0]
    z_right = boundary_conditions_z[1]
    r_left = boundary_conditions_r[0]
    r_right = boundary_conditions_r[1]
    if ptcl_pos_z >= z_left and ptcl_pos_z <= z_right and ptcl_pos_r >= r_left and ptcl_pos_r <= r_right:
        return True
    return False

@numba.njit(cache=True, fastmath=True)      
def filter_particles(pos_z, pos_r, Ntot, boundary_conditions_z, boundary_conditions_r):
    ptcl_idx = []
    for ip in range(Ntot):
        if internal_particles(pos_z[ip], pos_r[ip], boundary_conditions_z,
                                  boundary_conditions_r) == True:
            ptcl_idx.append(ip)
    return ptcl_idx
        

@numba.njit(cache=True, fastmath=True)
def CrossProduct(v1, v2):
    # можно использовать np.cross(a, b) !!!!!!!!!
    r = np.zeros(3)
    r[0] = v1[1]*v2[2]-v1[2]*v2[1]
    r[1] = -1*v1[0]*v2[2]+v1[2]*v2[0]
    r[2] = v1[0]*v2[1]-v1[1]*v2[0]
    return r

@numba.njit(cache=True, fastmath=True)
def UpdateSingleVelocityBoris(vel_z, vel_r, vel_y, Ez, Er, Ey,
                              Bz, Br, By, dt, q, m):
    #t = np.zeros(3)
    '''
    v_minus = np.zeros(3)
    v_prime = np.zeros(3)
    v_plus = np.zeros(3)
    v_minus_cross_t = np.zeros(3)
    v_prime_cross_s = np.zeros(3)
    '''
    tz = q/m*Bz*0.5*dt
    tr = q/m*Br*0.5*dt
    ty = q/m*By*0.5*dt
    t_mag2 = tz*tz + tr*tr + ty*ty
    sz = 2*tz/(1+t_mag2)
    sr = 2*tr/(1+t_mag2)
    sy = 2*ty/(1+t_mag2)
    # v_minus
    '''
    v_minus[0] = vel_z + q/m*Ez*0.5*dt
    v_minus[1] = vel_r + q/m*Er*0.5*dt
    v_minus[2] = vel_y + q/m*Ey*0.5*dt
    '''
    v_minus_z = vel_z + q/m*Ez*0.5*dt
    v_minus_r = vel_r + q/m*Er*0.5*dt
    v_minus_y = vel_y + q/m*Ey*0.5*dt
    # v_prime
    #v_minus_cross_t = CrossProduct(v_minus, t)
    '''
    v_minus_cross_t[0] = v_minus[1]*ty-v_minus[2]*tr
    v_minus_cross_t[1] = -1*v_minus[0]*ty+v_minus[2]*tz
    v_minus_cross_t[2] = v_minus[0]*ty-v_minus[1]*tz
    v_prime[0] = v_minus[0] + v_minus_cross_t[0]
    v_prime[1] = v_minus[1] + v_minus_cross_t[1]
    v_prime[2] = v_minus[2] + v_minus_cross_t[2]
    '''
    v_minus_cross_t_z = v_minus_r*ty-v_minus_y*tr
    v_minus_cross_t_r = -1*v_minus_z*ty+v_minus_y*tz
    v_minus_cross_t_y = v_minus_z*ty-v_minus_r*tz
    v_prime_z = v_minus_z + v_minus_cross_t_z
    v_prime_r = v_minus_r + v_minus_cross_t_r
    v_prime_y = v_minus_y + v_minus_cross_t_y
    # v_plus
    #v_prime_cross_s = CrossProduct(v_prime, s)
    '''
    v_prime_cross_s[0] = v_prime[1]*sy-v_prime[2]*sr
    v_prime_cross_s[1] = -1*v_prime[0]*sy+v_prime[2]*sz
    v_prime_cross_s[2] = v_prime[0]*sr-v_prime[1]*sz
    v_plus = v_minus + v_prime_cross_s
    '''
    v_prime_cross_s_z = v_prime_r*sy-v_prime_y*sr
    v_prime_cross_s_r = -1*v_prime_z*sy+v_prime_y*sz
    v_prime_cross_s_y = v_prime_z*sr-v_prime_r*sz
    v_plus_z = v_minus_z + v_prime_cross_s_z
    v_plus_r = v_minus_r + v_prime_cross_s_r
    v_plus_y = v_minus_y + v_prime_cross_s_y
    # vel n+1/2
    vel_z = v_plus_z + q/m*Ez*0.5*dt
    vel_r = v_plus_r + q/m*Er*0.5*dt
    vel_y = v_plus_y + q/m*Ey*0.5*dt
    return (vel_z, vel_r, vel_y)

@numba.njit(cache=True)
def UpdateVelocity(vel_z, vel_r, vel_y, Ez, Er, Ey, Bz, Br, By, dt, q, m, Ntot):
    # Loop over particles
    for ip in xrange(Ntot):
        vel_z[ip], vel_r[ip], vel_y[ip] = UpdateSingleVelocityBoris(vel_z[ip],
             vel_r[ip], vel_y[ip], Ez[ip], Er[ip], Ey[ip], Bz[ip], Br[ip], By[ip], dt, q, m)


@numba.njit(cache=True, fastmath=True)
def UpdatePosition(pos_z, pos_r, vel_z, vel_r, vel_y, dt, Ntot, boundary_conditions_z, 
                       boundary_conditions_r, internal_ptcls_idx):
    #internal_ptcl_list = []
    #internal_ptcl_list = []
    # Loop over particles
    for ip in prange(Ntot):
        pos_z[ip] += vel_z[ip]*dt
        pos_r[ip] += vel_r[ip]*dt
        pos_y_ip = 0.
        pos_y_ip += vel_y[ip]*dt
        #pos_y[ip] += vel_y[ip]*dt   
        #rotate particle back to ZR plane
        #r = math.sqrt(pos_r[ip]*pos_r[ip] + pos_y[ip]*pos_y[ip])
        r = math.sqrt(pos_r[ip]*pos_r[ip] + pos_y_ip*pos_y_ip)
        #sin_theta_r = pos_y[ip] / r
        sin_theta_r = pos_y_ip / r
        pos_r[ip] = r
        #pos_y[ip] = 0
        pos_y_ip = 0.
        #rotate velocity
        cos_theta_r = math.sqrt(1-sin_theta_r*sin_theta_r)
        u2 = cos_theta_r*vel_r[ip] - sin_theta_r*vel_y[ip]
        v2 = sin_theta_r*vel_r[ip] + cos_theta_r*vel_y[ip]
        vel_r[ip] = u2
        vel_y[ip] = v2
        '''
        if internal_particles(pos_z[ip], pos_r[ip], boundary_conditions_z, boundary_conditions_r)==True:
            internal_ptcls_idx[ip] = True
            #internal_ptcl_list.append(ip)
        else:
            internal_ptcls_idx[ip] = False
            #internal_ptcl_list.append(ip)
        '''
    
'''
@numba.njit(cache=True, nogil=True)
def Rotate_to_RZ_plane(pos_r, pos_y, vel_r, vel_y, Ntot):
    # Loop over particles
    for ip in numba.prange(Ntot):
        #rotate particle back to ZR plane
        r = math.sqrt(pos_r[ip]*pos_r[ip] + pos_y[ip]*pos_y[ip])
        sin_theta_r = pos_y[ip] / r
        pos_r[ip] = r
        pos_y[ip] = 0
        #rotate velocity
        cos_theta_r = math.sqrt(1-sin_theta_r*sin_theta_r)
        u2 = cos_theta_r*vel_r[ip] - sin_theta_r*vel_y[ip]
        v2 = sin_theta_r*vel_r[ip] + cos_theta_r*vel_y[ip]
        vel_r[ip] = u2
        vel_y[ip] = v2
'''
#@numba.njit(cache=True)
def ParticlePush(pos_z, pos_r, vel_z, vel_r, vel_y, Ez, Er, Ey, 
                       Bz, Br, By, dt, q, m, Ntot, 
                       ptcl_boundary_conditions_z, 
                       ptcl_boundary_conditions_r, internal_ptcls_idx):
    # push velocity back in time by 1/2 dt
    #UpdateVelocity(vel_z, vel_r, vel_y, Ez, Er, Ey, Bz, Br, By, -0.5*dt, q, m, Ntot)
    #t0 = time.time()
    
    UpdateVelocity(vel_z, vel_r, vel_y, Ez, Er, Ey, Bz, Br, By,
                       dt, q, m, Ntot)
    
    UpdatePosition(pos_z, pos_r, vel_z, vel_r, vel_y, dt, Ntot, 
             ptcl_boundary_conditions_z, ptcl_boundary_conditions_r, 
             internal_ptcls_idx)
    '''
    internal_ptcls_idx = filter_particles(pos_z, pos_r, Ntot, ptcl_boundary_conditions_z, 
                                          ptcl_boundary_conditions_r)
    
    print len(internal_ptcls_idx)
    '''
    #return ptcl_idx
    #Rotate_to_RZ_plane(pos_r, pos_y, vel_r, vel_y, Ntot)
    #print(time.time() - t0, "End of iteration")

import cProfile
    
def main():
    Ntot = int(1e6)
    pos_z = np.random.random_sample(size=Ntot)
    pos_r = np.random.random_sample(size=Ntot)
    pos_y = np.zeros(Ntot)
    vel_z = np.random.random_sample(size=Ntot)
    vel_r = np.random.random_sample(size=Ntot)
    vel_y = np.random.random_sample(size=Ntot)
    dt = 1e-10
    q = e
    m = m_e
    Ez = np.random.sample(Ntot)
    Er = np.random.sample(Ntot)
    Ey = np.random.sample(Ntot)
    Bz = np.random.sample(Ntot)
    Br = np.random.sample(Ntot)
    By = np.random.sample(Ntot)
    #Ez, Er, Ey = np.array([np.random.sample(Ntot), np.random.sample(Ntot), np.zeros(Ntot)]).T
    #Bz, Br, By = np.array([np.random.sample(Ntot), np.zeros(Ntot), np.zeros(Ntot)]).T
    Num_iter = 1
    boundary_conditions_z = np.array([0, 0.5])
    boundary_conditions_r = np.array([0, 0.5])
    internal_ptcls_idx = np.empty(Ntot, dtype=bool)
    emission_ptcls_idx = [1]
    print "init:", pos_z.shape[0]
    ParticlePush(pos_z, pos_r, vel_z, vel_r, vel_y, Ez, Er, Ey, 
            Bz, Br, By, dt, q, m, Ntot, boundary_conditions_z, 
            boundary_conditions_r, internal_ptcls_idx)
    print internal_ptcls_idx[internal_ptcls_idx==True].shape[0]
    #print internal_ptcls_idx[internal_ptcls_idx==False].shape
    summ_t = 0
    #print pos_z[0]
    for _ in xrange(Num_iter):
        t0 = time.time()
        ParticlePush(pos_z, pos_r, vel_z, vel_r, vel_y, Ez, Er, Ey, 
                Bz, Br, By, dt, q, m, Ntot, boundary_conditions_z,
                boundary_conditions_r, internal_ptcls_idx)
        t1 = time.time()
        summ_t += t1 - t0
    print summ_t
    #print pos_z[0]
    
if __name__ == '__main__':
    main()
    #cProfile.run('main()', sort='tottime')
    