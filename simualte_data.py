# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 17:08:19 2021

@author: user
"""

import os
import numpy as np
import random
import pickle

subjects = 40
nodes = 35
views = 4

mu_0 = 3
sigma_0 = 0.2
mu_1 = 2.8
sigma_1 = 0.3

def simulate_data(subjects, nodes, views, sigma, mu):
    edges = int(nodes*(nodes-1)/2)
    adjs = []
    for subject in range(subjects):
        dist_mat = np.zeros((nodes,nodes,views)) # Initialize nxn matrix
        for view in range(views):
            dist_arr = np.random.normal(mu, sigma, edges)
            dist_list = dist_arr.tolist()
            k = 0
            for i in range(nodes):
                for j in range(nodes):
                    if i>j:
                        dist_mat[i,j,view] = dist_list[k]
                        dist_mat[j,i,view] = dist_mat[i,j,view]
                        k+=1
        adjs.append(dist_mat)
    return adjs
adjs_0 = simulate_data(subjects, nodes, views, sigma_0, mu_0)
adjs_1 = simulate_data(subjects, nodes, views, sigma_1, mu_1)
adjs = adjs_0 + adjs_1
labels = [0] * subjects + [1] * subjects
with open('data/Demo/Demo_edges', 'wb') as f:
    pickle.dump(adjs, f)
with open('data/Demo/Demo_labels', 'wb') as f:
    pickle.dump(labels, f)
print("s")