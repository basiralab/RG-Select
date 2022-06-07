# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:44:19 2021

@author: Mohammed Amine
"""
import pickle
import math
import os
import torch
import random
import numpy as np
from torch_geometric import utils
from torch_geometric.data import Data

# Splits the dataset into k-folds  
def stratify_splits(graphs, n_fold):
    graphs_0 = []
    graphs_1 = []
    for i in range(len(graphs)):
        if graphs[i]['label'] == 0:
            graphs_0.append(graphs[i])
        if graphs[i]['label'] == 1:
            graphs_1.append(graphs[i])
    graphs_0_folds = []
    graphs_1_folds = []
    pop_0_fold_size = math.floor(len(graphs_0) / n_fold)
    pop_1_fold_size = math.floor(len(graphs_1) / n_fold)
    graphs_0_folds = [graphs_0[i:i + pop_0_fold_size] for i in range(0, len(graphs_0), pop_0_fold_size)]
    graphs_1_folds = [graphs_1[i:i + pop_1_fold_size] for i in range(0, len(graphs_1), pop_1_fold_size)]
    folds = []
    for i in range(n_fold):
        fold = []
        fold.extend(graphs_0_folds[i])
        fold.extend(graphs_1_folds[i])
        folds.append(fold)
    
    if len(graphs_0_folds) > n_fold:
        folds[n_fold-1].extend(graphs_0_folds[n_fold])
    if len(graphs_1_folds) > n_fold:
        folds[n_fold-1].extend(graphs_1_folds[n_fold])

    return folds

# Saves train and test sets.
def split_data_fold(n_fold, dataset):
    
    if not os.path.exists('Folds'+str(n_fold)):
        os.makedirs('Folds'+str(n_fold))
      
    with open('data/'+dataset+'/'+dataset+'_edges','rb') as f:
        multigraphs = pickle.load(f)        
    with open('data/'+dataset+'/'+dataset+'_labels','rb') as f:
        labels = pickle.load(f)
        
    G_list = []
    for i in range(len(labels)):
        G_element = {"adj": multigraphs[i],"label": labels[i],"id":  i,}
        G_list.append(G_element)
            
    folds = stratify_splits(G_list, n_fold)
    
    [random.shuffle(folds[i]) for i in range(len(folds))]
    for i in range(len(folds)):
        test_folds = []
        train_folds = []
        test_folds.extend(folds[i])
        for j  in range(len(folds)):
            if j==i :
                continue
            else : 
                train_folds.extend(folds[j])
        with open('Folds'+str(n_fold)+'/'+'Folds_'+str(n_fold)+'_'+dataset+'_fold_'+str(i)+'_train', 'wb') as f:
            pickle.dump(train_folds, f)
        with open('Folds'+str(n_fold)+'/'+'Folds_'+str(n_fold)+'_'+dataset+'_fold_'+str(i)+'_test', 'wb') as f:
            pickle.dump(test_folds, f) 

# Splits and saves views of train and test sets
def split_views(n_fold, dataset):
    
    if not os.path.exists('Folds_views'+str(n_fold)):
        os.makedirs('Folds_views'+str(n_fold))

    rep = 'Folds'+str(n_fold)+'/'
    dest = 'Folds_views'+str(n_fold)+'/'
    #dest = 'Folds_'+str(n_fold)+'_views'+str(n_fold)+'/'
    
    for i in range(n_fold):
        with open(rep + 'Folds_'+str(n_fold) +'_'+ dataset+'_fold_'+str(i)+'_train','rb') as f:
            G_list_train_i = pickle.load(f)
        with open(rep + 'Folds_'+str(n_fold) +'_'+ dataset+'_fold_'+str(i)+'_test','rb') as f:
            G_list_test_i = pickle.load(f)
        
        n_views = G_list_train_i[0]['adj'].shape[2]
        for v in range(n_views):
            with open(rep + 'Folds_'+str(n_fold) +'_'+ dataset+'_fold_'+str(i)+'_train','rb') as f:
                G_list_train_i = pickle.load(f)
            with open(rep + 'Folds_'+str(n_fold) +'_'+ dataset+'_fold_'+str(i)+'_test','rb') as f:
                G_list_test_i = pickle.load(f)
            G_list_train_i_view_v = G_list_train_i
            G_list_test_i_view_v = G_list_test_i
            
            for j in range(len(G_list_train_i)):
                G_list_train_i_view_v[j]['adj'] = G_list_train_i[j]['adj'][:,:,v]
                
            for k in range(len(G_list_test_i)):
                G_list_test_i_view_v[k]['adj'] = G_list_test_i[k]['adj'][:,:,v]
 
        
            with open(dest + dataset + '_view_'+str(v)+'_folds_'+ str(n_fold) + '_fold_' + str(i) +'_train','wb') as f:
                pickle.dump(G_list_train_i_view_v, f)
            with open(dest + dataset + '_view_'+str(v)+'_folds_'+ str(n_fold) + '_fold_' + str(i) + '_test','wb') as f:
                pickle.dump(G_list_test_i_view_v, f)
   
# Transform train and test sets into pytorch-geometric Data.
def transform(n_fold, dataset):
        
    dest = 'Folds_views'+str(n_fold)+'/'   
    if not os.path.exists('Folds_processed'+str(n_fold)):
        os.makedirs('Folds_processed'+str(n_fold))
    
    for cv in range(n_fold):        
        with open('Folds'+str(n_fold)+'/'+'Folds_'+str(n_fold)+'_'+dataset+'_fold_'+str(cv)+'_train','rb') as f:
            G_list_train_i = pickle.load(f)
            
        n_views = G_list_train_i[0]['adj'].shape[2]
        
        for v in range(n_views):
            train_list_pg = []
            test_list_pg = []
            with open(dest + dataset + '_view_'+str(v) +'_folds_'+ str(n_fold) + '_fold_' + str(cv) +'_train','rb') as f:
                list_train = pickle.load(f)
            with open(dest + dataset + '_view_'+str(v) +'_folds_'+ str(n_fold) + '_fold_' + str(cv) +'_test','rb') as f:
                list_test = pickle.load(f)
            for i in range(len(list_train)):
                adj = torch.from_numpy(list_train[i]['adj'])
                edge_index, edge_values = utils.dense_to_sparse(adj)
                x = torch.eye(adj.shape[0])
                data_train_elt = Data(x=x, edge_index=edge_index, edge_attr=edge_values, adj=adj, y=torch.tensor([list_train[i]['label']]))
                train_list_pg.append(data_train_elt)
            for j in range(len(list_test)):
                adj = torch.from_numpy(list_test[j]['adj'])
                edge_index, edge_values = utils.dense_to_sparse(adj)
                x = torch.eye(adj.shape[0])
                data_test_elt = Data(x=x, edge_index=edge_index, edge_attr=edge_values, adj=adj, y=torch.tensor([list_test[j]['label']]))
                test_list_pg.append(data_test_elt)
            
            with open('Folds_processed'+str(n_fold)+'/'+dataset+'_view_'+str(v) +'_folds_'+ str(n_fold) + '_fold_'+str(cv)+'_train_pg','wb') as f:
                pickle.dump(train_list_pg, f)
            with open('Folds_processed'+str(n_fold)+'/'+dataset+'_view_'+str(v) +'_folds_'+ str(n_fold) + '_fold_'+str(cv)+'_test_pg','wb') as f:
                pickle.dump(test_list_pg, f)  
                
# Saves the training and test sets of 5 folds cross validation.
def transform_Data(n_fold, dataset):   
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)    
    
    if not os.path.exists('Folds_processed'+str(n_fold)+'/'+dataset+'_view_'+str(0)+'_folds_'+ str(n_fold) +'_fold_'+str(0)+'_test_pg'):
        split_data_fold(n_fold, dataset)
        split_views(n_fold, dataset)
        transform(n_fold, dataset)




