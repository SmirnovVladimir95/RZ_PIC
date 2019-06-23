#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:38:18 2019

@author: vladimirsmirnov
"""
import numpy as np

class NumpyDynamicArray1D(object):
    def __init__(self, size, dtype=float, multy_coeff = 5):
        self.massive = np.empty(size*multy_coeff, dtype)
        self.size = size
        self.multy_coeff = multy_coeff
        self.type = dtype
    
    def append(self, item):
        try:
            self.massive[self.size] = item
        except IndexError:
            old_massive = self.massive[:self.size]
            self.massive = np.empty(self.size*self.multy_coeff, dtype=self.type)
            self.massive[:self.size] = old_massive
            self.massive[self.size] = item
        self.size += 1
        
    def extend(self, massive):
        try:
            self.massive[self.size:self.size+massive.shape[0]] = massive
        except ValueError:
            flag = False
            while not flag:
                flag = self._memory_allocation(massive)
        self.size += massive.shape[0]
    
    def array_get(self):
        return self.massive[:self.size]
    
    def array_size(self):
        return self.size
    
    def array_set(self, massive=None, idx=None, value=None):
        #assert len(self.array_get()) == len(massive), "wrong array size"
        #assert type(massive) == np.ndarray
        if idx is not None:
            self.massive[idx] = value
        else:
            self.massive = np.copy(massive)
            self.size = massive.shape[0]
        
    def _memory_allocation(self, massive):
        print "Warning: multy_coeff should be increased"
        try:
            old_massive = self.massive[:self.size]
            self.massive = np.empty(self.size*self.multy_coeff, dtype=self.type)
            self.massive[:self.size] = old_massive
            self.massive[self.size:self.size+len(massive)]
            self.massive[self.size:self.size+len(massive)] = massive
        except ValueError:
            self.old_multy_coeff = self.multy_coeff
            self.multy_coeff = np.round(massive.shape[0] / self.multy_coeff)
            return False
        else:
            self.multy_coeff = self.old_multy_coeff
            del self.old_multy_coeff
            return True
        
if __name__ == '__main__':
    #a = NumpyDynamicArray1D()
    pass