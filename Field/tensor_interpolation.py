#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:31:17 2019

@author: vladimirsmirnov
"""
import scipy.interpolate
import numpy as np
import math

def grid_coarse(grid, coeff):
    size = int(math.ceil(grid.shape[0]/float(coeff)))
    grid_new = np.zeros(size)
    count = 0
    for idx, item in enumerate(grid):
        if idx % coeff == 0:
            grid_new[count] = item
            count += 1
    if grid_new[-1] == grid[-1]:
        return grid_new
    else:
        return np.concatenate((grid_new, grid[-1:]), axis=0) 
    
def grid_fine(grid, coeff):
    assert grid.shape[0] > 2, "Too small grid"
    size1 = (grid.shape[0]-1)*coeff - coeff + 1
    grid1 = np.linspace(0, grid[-2], size1, dtype=int)
    grid2 = np.arange(grid[-2], grid[-1], grid1[1]-grid1[0])
    return np.concatenate((grid1, grid2[1:], grid[-1:]), axis=0)

def tensor_func(tensor, grid):
    values = np.zeros(grid[0].shape[0]*grid[1].shape[0])
    points = np.zeros((grid[0].shape[0]*grid[1].shape[0], 2))
    k = 0
    for i in range(len(grid[0])):
        for j in range(len(grid[1])):
            values[k] = tensor[i][j]
            points[k][0] = grid[0][i]
            points[k][1] = grid[1][j]
            k += 1
    return values, points

def tensor_interpolation(tensor, coeff, grid=None, Type=None, method='linear'):
    assert len(tensor.shape) <= 2, "tensor.shape must be 1D or 2D"
    if Type == "coarse":
        if len(tensor.shape) == 1:
            if grid is None:
                grid = np.arange(0, tensor.shape[0], 1)
            values, points = tensor, grid
            grid_new = grid_coarse(grid, coeff)
            f = scipy.interpolate.interp1d(points, values,  kind=method)
            tensor_interp = f(grid_new)
            return tensor_interp, (grid_new)
        if len(tensor.shape) == 2:
            if grid is None:
                grid = []
                for idx in range(len(tensor.shape)):
                    grid.append(np.arange(0, tensor.shape[idx], 1))
            values, points = tensor_func(tensor, grid)
            grid_z_new = grid_coarse(grid[0], coeff)
            grid_r_new = grid_coarse(grid[1], coeff)
            grid_new = np.meshgrid(grid_z_new, grid_r_new)
            tensor_interp = scipy.interpolate.griddata(points, values, (grid_new[0], grid_new[1]), 
                                            method=method)
            return tensor_interp.T, (grid_z_new, grid_r_new)
    if Type == "fine":
        if len(tensor.shape) == 1:
            values, points = tensor, grid
            grid_new = grid_fine(grid, coeff)
            f = scipy.interpolate.interp1d(points, values, kind=method)
            tensor_interp = f(grid_new)
            return tensor_interp, (grid_new)
        if len(tensor.shape) == 2:
            if grid is None:
                grid = []
                for idx in range(len(tensor.shape)):
                    grid.append(np.arange(0, tensor.shape[idx], 1))
            values, points = tensor_func(tensor, grid)
            grid_z_new = grid_fine(grid[0], coeff)
            grid_r_new = grid_fine(grid[1], coeff)
            grid_new = np.meshgrid(grid_z_new, grid_r_new)
            tensor_interp = scipy.interpolate.griddata(points, values, (grid_new[0], grid_new[1]), 
                                            method=method)
            return tensor_interp.T, (grid_z_new, grid_r_new)
    
    return "wrong Type"
    

    
