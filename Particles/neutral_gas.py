#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:30:35 2019

@author: vladimirsmirnov
"""

import numpy as np
from scipy.constants import k
import math

class NeutralGas:
    def __init__(self, T, n, mass):
        self.T = T
        self.n = n
        self.mass = mass
    
    def gen_vel_vector(self):
        sigma = math.sqrt(k*self.T/self.mass)
        return sigma * np.random.randn(3)