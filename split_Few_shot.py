# -*- coding: utf-8 -*-

import pickle
import os
import numpy as np
import random
import torch

from torch_geometric import utils
from torch_geometric.data import Data

# Checks if selected subset is used before.
def check_new_samples(new_samples, trained_set):
    
    if new_samples[0]==new_samples[1]:
        return True
    elif new_samples[2]==new_samples[3]:
        return True
    
    for i in trained_set:
        same_samples = 0
        
        for k in range(4):
            if new_samples[k] == i[k]:
                same_samples+=1
                
        if new_samples[0] == i[1]:
            same_samples+=1
        if new_samples[1] == i[0]:
            same_samples+=1
        if new_samples[2] == i[3]:
            same_samples+=1
        if new_samples[3] == i[2]:
            same_samples+=1
                
        if same_samples >1:
            return True
        
    return False

# Randomly selects a training set of 2 samples from each class.
def two_shot_split(graphs, args,num_samples, trained_set):
    graphs_0 = []
    graphs_1 = []
    for i in range(len(graphs)):
        if graphs[i]['label'] == 0:
            graphs_0.append(graphs[i])
        if graphs[i]['label'] == 1:
            graphs_1.append(graphs[i])
    train = []
    rand_integers = [random.randint(0,len(graphs_0)-1), random.randint(0,len(graphs_0)-1), random.randint(0,len(graphs_1)-1), random.randint(0,len(graphs_1)-1)]
    while check_new_samples(rand_integers,trained_set):
      rand_integers = [random.randint(0,len(graphs_0)-1), random.randint(0,len(graphs_0)-1), random.randint(0,len(graphs_1)-1), random.randint(0,len(graphs_1)-1)]
    print("*"*50)
    print("Indexes:"+str(rand_integers[0])+" "+str(rand_integers[1])+" "+str(rand_integers[2])+" "+str(rand_integers[3])+" ")
    for i in range(0,num_samples*2):
      if i<num_samples:
        train.append(graphs_0[rand_integers[i]])
      else:
        train.append(graphs_1[rand_integers[i]])
    
    graphs_0.pop(rand_integers[0])
    if rand_integers[0]>rand_integers[1]:
      graphs_0.pop(rand_integers[1])
    else:
      graphs_0.pop(rand_integers[1]-1)
    
    graphs_1.pop(rand_integers[2])
    if rand_integers[2]>rand_integers[3]:
      graphs_1.pop(rand_integers[3])
    else:
      graphs_1.pop(rand_integers[3]-1)

    val = []
    val.extend(graphs_0)
    val.extend(graphs_1)
    test = []
    return train, val, test, rand_integers


# Randomly selects 100 different training and test sets
def few_shot_splits(dataset):
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    used_indexes = []
    if not os.path.exists('Two_shot_samples'):
        os.makedirs('Two_shot_samples')
      
    
    with open('data/'+dataset+'/'+dataset+'_edges','rb') as f:
        multigraphs = pickle.load(f)        
    with open('data/'+dataset+'/'+dataset+'_labels','rb') as f:
        labels = pickle.load(f)
    for shot_n in range(100):  
        
        G_list = []
        for i in range(len(labels)):
            G_element = {"adj": multigraphs[i],"label": labels[i],"id":  i,}
            G_list.append(G_element)

        np.random.shuffle(G_list)
        size_label0=0
        size_label1=0
        for i in range(len(G_list)):
            if G_list[i]['label'] == 0:
                size_label0+=1
            if G_list[i]['label'] == 1:
                size_label1+=1
                
        if not (size_label1>=25 and size_label0>=25):
            print(size_label1)
            print(size_label0)
            print("There must be at least 25 samples from each class.")
            exit()
        
        args = ""
        
        train, val, test, rand_integers = two_shot_split(G_list, args,2, used_indexes)
        used_indexes.append(rand_integers)

        
        test_folds = []
        train_folds = []
        test_folds.extend(val)
        train_folds.extend(train)
        
        with open('Two_shot_samples/'+dataset+'_Two_Shot_'+str(shot_n)+'_train', 'wb') as f:
            pickle.dump(train_folds, f)
        with open('Two_shot_samples/'+dataset+'_Two_Shot_'+str(shot_n)+'_test', 'wb') as f:
            pickle.dump(test_folds, f)

    with open('Two_shot_samples/indexes_'+dataset, 'wb') as f:
                pickle.dump(used_indexes, f)
    used_indexes = []
    
# Splits the views of selected training and test sets.    
def few_shot_split_views(dataset):
    if not os.path.exists('Two_shot_samples_views'):
        os.makedirs('Two_shot_samples_views')
    rep = 'Two_shot_samples/'
    dest = 'Two_shot_samples_views/'
    
    for shot_n in range(100):
        with open(rep +dataset+'_Two_Shot_'+str(shot_n)+'_train','rb') as f:
            G_list_train_i = pickle.load(f)
        with open(rep +dataset+'_Two_Shot_'+str(shot_n)+'_test','rb') as f:
            G_list_test_i = pickle.load(f)
        
        n_views = G_list_train_i[0]['adj'].shape[2]
        for v in range(n_views):
            with open(rep +dataset+'_Two_Shot_'+str(shot_n)+'_train','rb') as f:
                G_list_train_i = pickle.load(f)
            with open(rep +dataset+'_Two_Shot_'+str(shot_n)+'_test','rb') as f:
                G_list_test_i = pickle.load(f)
            G_list_train_i_view_v = G_list_train_i
            G_list_test_i_view_v = G_list_test_i
            
            for j in range(len(G_list_train_i)):
                G_list_train_i_view_v[j]['adj'] = G_list_train_i[j]['adj'][:,:,v]
                
            for k in range(len(G_list_test_i)):
                G_list_test_i_view_v[k]['adj'] = G_list_test_i[k]['adj'][:,:,v]
            
            with open(dest + dataset + '_view_'+str(v)+ '_shot_' + str(shot_n) +'_train','wb') as f:
                pickle.dump(G_list_train_i_view_v, f)
            with open(dest + dataset + '_view_'+str(v)+ '_shot_' + str(shot_n) + '_test','wb') as f:
                pickle.dump(G_list_test_i_view_v, f)


# Transform train and test sets into pytorch-geometric Data.
def few_shot_transformer(dataset):
    rep = 'Two_shot_samples/'   
    dest = 'Two_shot_samples_views/'   
    
    if not os.path.exists('Two_shot_processed'):
        os.makedirs('Two_shot_processed')
    
    for shot_n in range(100):
        
        with open(rep +dataset+'_Two_Shot_'+str(0)+'_train','rb') as f:
            G_list_train_i = pickle.load(f)
        
        n_views = G_list_train_i[0]['adj'].shape[2]
        
        for v in range(n_views):
            train_list_pg = []
            test_list_pg = []
            with open(dest + dataset + '_view_'+str(v)+ '_shot_' + str(shot_n) +'_train','rb') as f:
                list_train = pickle.load(f)
            with open(dest + dataset + '_view_'+str(v)+ '_shot_' + str(shot_n) +'_test','rb') as f:
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
            
            with open('Two_shot_processed/'+dataset+'_view_'+str(v)+'_shot_'+str(shot_n)+'_train_pg','wb') as f:
                pickle.dump(train_list_pg, f)
            with open('Two_shot_processed/'+dataset+'_view_'+str(v)+'_shot_'+str(shot_n)+'_test_pg','wb') as f:
                pickle.dump(test_list_pg, f)

# Saves the training and test sets of Two shot learning.
def transform_Data_FewShot(dataset):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  
    if not os.path.exists('Two_shot_processed/'+dataset+'_view_'+str(0)+'_shot_'+str(0)+'_train_pg'):
        few_shot_splits(dataset)
        few_shot_split_views(dataset)
        few_shot_transformer(dataset)
    
            



