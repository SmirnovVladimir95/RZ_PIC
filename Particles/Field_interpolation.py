#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:21:58 2019

@author: vladimirsmirnov
"""

def gather(data,lc):
    i = math.trunc(lc[0])
    j = math.trunc(lc[1])
    di = lc[0] - i
    dj = lc[1] - j
    return  (data[i][j]*(1-di)*(1-dj) +
          data[i+1][j]*(di)*(1-dj) + 
          data[i][j+1]*(1-di)*(dj) + 
          data[i+1][j+1]*(di)*(dj)) 
    
def scatter(data,lc,value):
    i = int(numpy.trunc(lc[0]))
    j = int(numpy.trunc(lc[1]))
    di = lc[0] - i
    dj = lc[1] - j
            
    data[i][j] += (1-di)*(1-dj)*value
    data[i+1][j] += (di)*(1-dj)*value
    data[i][j+1] += (1-di)*(dj)*value
    data[i+1][j+1] += (di)*(dj)*value