# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:52:58 2021

@author: Mohammed Amine
"""

import pickle
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import torch
#import region_labels

def extract_weights_single(dataset, view, model, training_type, shot_n, cv_n):
    if model == 'sag':
        fs_path = '{}/weights/W_{}_{}_{}_view_{}_{}.pickle'.format(model, training_type, dataset, model, view, shot_n)
        cv_path = '{}/weights/W_MainModel_{}_{}_{}_view_{}_CV_{}.pickle'.format(model,training_type, dataset, model, view, cv_n)
    else:
        fs_path = '{}/weights/W_{}_{}_{}{}_view_{}.pickle'.format(model, training_type, dataset, model, shot_n, view)
        cv_path = '{}/weights/W_MainModel_{}_{}_{}_CV_{}_view_{}.pickle'.format(model,training_type, dataset, model, cv_n, view)
    if training_type == 'Few_Shot':
        x_path = fs_path
    else: 
        x_path = cv_path 
    with open(x_path,'rb') as f:
        weights = pickle.load(f)
    if model == 'sag':
        weights_vector = torch.mean(weights['w'], 1).detach().numpy()
    if model == 'diffpool':
        weights_vector = torch.mean(weights['w'], 1).detach().numpy()
    if model == 'gcn':
        weights_vector = weights['w'].squeeze().detach().numpy()
    if model == 'gat':
        weights_vector = weights['w'].squeeze().detach().numpy()
    if model == 'gunet':
        weights_vector = torch.mean(weights['w'], 0).detach().numpy()    
    return weights_vector

def extract_weights(dataset, view, model, training_type):
    runs = []
    if training_type == 'Few_Shot':
        for shot_i in range(5):
            runs.append(extract_weights_single(dataset, view, model, training_type, shot_i, 0))
    if training_type == '3Fold':
        for cv_i in range(3):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i))
    if training_type == '5Fold':
        for cv_i in range(5):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i))
    if training_type == '10Fold':
        for cv_i in range(10):
            runs.append(extract_weights_single(dataset, view, model, training_type, 0, cv_i))
    runs = np.array(runs)
    weights = np.mean(runs, axis=0)
    return weights

def top_biomarkers(weights, K_i):
    weights_normalized = np.abs(weights)
    result = []
    w_sorted = weights_normalized.argsort()  #verified
    for i in range(1, 1+K_i):
        result.append(w_sorted[-1*i])
    return result

def sim(nodes1, nodes2):
    if len(nodes1)==len(nodes2):
        counter = 0
        for i in nodes1:
            for k in nodes2:
                if i==k:
                    counter+=1
        return counter/len(nodes1)
    else:
        print('nodes vectors are not caompatible')

def sim_respective(nodes1, nodes2):
    if len(nodes1)==len(nodes2):
        counter = 0
        for i in range(len(nodes1)):
            if nodes1[i]==nodes2[i]:
                counter+=1
        return counter/len(nodes1)
    else:
        print('nodes vectors are not caompatible')
        
def sim_respective_weighted(rank1, rank2, strength1, strength2): # ongoing
    if len(rank1)==len(rank2) and len(strength1) == len(strength2) and len(rank1)==len(strength1):
        n_views = max(rank1)
        differences_rank =  np.abs(rank1 - rank2)
        differences_rank_weights = 1 - (differences_rank *1/n_views) 
        differences_strength = np.abs(strength1 - strength2)
        max_diff_strength = max(differences_strength)
        differences_strength_norm = differences_strength/max_diff_strength
        differences_strength_weights = 1 - differences_strength_norm
        
        sum_weights = np.sum(differences_rank_weights*differences_strength_weights)
        weighted_intersection = sum_weights/len(rank1)
        return weighted_intersection 
    else:
        print('nodes vectors are not caompatible')
        
def view_specific_rep(dataset,view,training_type, models):
    #models = ['diffpool', 'gat', 'gcn', 'gunet', 'sag']
    Ks = [5, 10, 15, 20]
    rep = np.zeros([len(models), len(models), len(Ks)])
    
    for i in range(rep.shape[0]):
        for j in range(rep.shape[1]):
            weights_i = extract_weights(dataset, view, models[i], training_type)
            weights_j = extract_weights(dataset, view, models[j], training_type)
            a=4
            for k in range(rep.shape[2]):
                top_bio_i = top_biomarkers(weights_i, Ks[k])
                top_bio_j = top_biomarkers(weights_j, Ks[k])
                rep[i,j,k] = sim(top_bio_i, top_bio_j)
    rep_mean = np.mean(rep, axis=2)
    rep_dict = {}
    rep_dict['matrix'] = rep_mean
    rep_dict['dataset'] = dataset
    rep_dict['view'] = view
    rep_dict['models'] = models
    rep_dict['training_type'] = training_type
    return rep_dict

def overall_avg_rep_cv_fixed(data_dict, training_type):
    dataset = data_dict['dataset']
    views = data_dict['views']
    models = data_dict['models']
    rep = np.zeros([len(models), len(models), len(views)])
    for view in views:
        rep_dict = view_specific_rep(dataset,view,training_type,models)
        rep[:,:,view] = rep_dict['matrix']
    rep_mean = np.mean(rep, axis=2)
    rep_dict = {}
    rep_dict['matrix'] = rep_mean
    rep_dict['models'] = models
    rep_dict['dataset'] = dataset
    rep_dict['training_type'] = training_type
    return rep_dict
        
def overall_avg_rep(data_dict):
    models = data_dict['models']
    dataset = data_dict['dataset']
    training_types = data_dict['training_types']
    rep = np.zeros([len(models), len(models), len(training_types)])
    for i in range(len(training_types)):
        if i ==2:
            s = 3
        rep_dict = overall_avg_rep_cv_fixed(data_dict, training_types[i])
        rep[:,:,i] = rep_dict['matrix']
    rep_mean = np.mean(rep, axis=2)
    rep_dict = {}
    rep_dict['matrix'] = rep_mean
    rep_dict['models'] = models
    rep_dict['dataset'] = dataset
    return rep_dict
    

def overall_avg_rep_plot(rep_dict, save_fig=False):
    models = rep_dict['models']
    df_cm = pd.DataFrame(rep_dict['matrix'], index = [i for i in models], columns = [i for i in models])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True ,vmin=0, vmax=1)
    title_msg = 'Overall average reproducibility Dataset: '+rep_dict['dataset']
    plt.title(title_msg)
    if save_fig==True:
        plt.savefig("./imgs/Rep_"+ rep_dict['dataset'] + '_avg'+".png")
    plt.show()
    plt.close()
    
def GNN_specific_rep_vect(dataset,views,training_type, model):
    #models = ['diffpool', 'gat', 'gcn', 'gunet', 'sag']
    Ks = [5, 10, 15, 20]
    rep = np.zeros([len(views), len(views), len(Ks)])
    
    for i in range(rep.shape[0]):
        for j in range(rep.shape[1]):
            weights_i = extract_weights(dataset, views[i], model, training_type)
            weights_j = extract_weights(dataset, views[j], model, training_type)
            for k in range(rep.shape[2]):
                top_bio_i = top_biomarkers(weights_i, Ks[k])
                top_bio_j = top_biomarkers(weights_j, Ks[k])
                rep[i,j,k] = sim(top_bio_i, top_bio_j)
    rep_mean = np.mean(rep, axis=2)
    rep_vec = np.sum(rep_mean, axis=1)
    rep_dict = {}
    rep_dict['strength_vector'] = rep_vec
    rep_dict['rank_vector'] = rep_vec.argsort()[::-1].argsort() # verified
    rep_dict['dataset'] = dataset
    rep_dict['views'] = views
    rep_dict['model'] = model
    rep_dict['training_type'] = training_type
    return rep_dict

def overall_corr_rep_cv_fixed(data_dict, training_type):
    dataset = data_dict['dataset']
    views = data_dict['views']
    models = data_dict['models']
    rep_rank = np.zeros([len(models), len(models)])
    rep_strength = np.zeros([len(models), len(models)])
    for i in range(len(models)):
        rep_vect_i = GNN_specific_rep_vect(dataset,views,training_type, models[i])
        rep_rank_i = rep_vect_i['rank_vector']
        rep_strength_i = rep_vect_i['strength_vector']
        for j in range(len(models)):
            rep_vect_j = GNN_specific_rep_vect(dataset,views,training_type, models[j])
            rep_rank_j = rep_vect_j['rank_vector']
            rep_strength_j = rep_vect_j['strength_vector']
            corr_rank = np.corrcoef(rep_rank_i, rep_rank_j)
            corr_strength = np.corrcoef(rep_strength_i, rep_strength_j)
            rep_rank[i,j] = corr_rank[0,1]
            rep_strength[i,j] = corr_strength[0,1] 
    #rep_mean = np.mean(rep, axis=2)
    rep_dict = {}
    rep_dict['rank_matrix'] = rep_rank
    rep_dict['strength_matrix'] = rep_strength
    rep_dict['models'] = models
    rep_dict['dataset'] = dataset
    rep_dict['training_type'] = training_type
    return rep_dict

def overall_corr_rep(data_dict):
    models = data_dict['models']
    dataset = data_dict['dataset']
    training_types = data_dict['training_types']
    rep_rank = np.zeros([len(models), len(models), len(training_types)])
    rep_strength = np.zeros([len(models), len(models), len(training_types)])
    for i in range(len(training_types)):
        rep_dict = overall_corr_rep_cv_fixed(data_dict, training_types[i])
        rep_rank[:,:,i] = rep_dict['rank_matrix']
        rep_strength[:,:,i] = rep_dict['strength_matrix']
    rep_rank_mean = np.mean(rep_rank, axis=2)
    rep_strength_mean = np.mean(rep_strength, axis=2)
    rep_dict = {}
    rep_dict['rank_matrix'] = rep_rank_mean
    rep_dict['strength_matrix'] = rep_strength_mean
    rep_dict['models'] = models
    rep_dict['dataset'] = dataset
    return rep_dict

def overall_corr_rep_plot(rep_dict, corr_type='rank', save_fig=False):
    models = rep_dict['models']
    corr_key = corr_type + '_matrix'
    df_cm = pd.DataFrame(rep_dict[corr_key], index = [i for i in models], columns = [i for i in models])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True ,vmin=0, vmax=1)
    title_msg = 'Overall '+ corr_type +' correlation reproducibility Dataset: '+rep_dict['dataset']  
    plt.title(title_msg)
    if save_fig==True:
        plt.savefig("./imgs/Rep_"+ rep_dict['dataset'] + '_' +corr_type + ".png")
    plt.show()
    plt.close()  
    
def GNN_specific_rep_accumulated_vect(dataset,views,training_type, model):
    #models = ['diffpool', 'gat', 'gcn', 'gunet', 'sag']
    Ks = [5, 10, 15, 20]
    rep = np.zeros([len(views), len(views), len(Ks)])
    
    for i in range(rep.shape[0]):
        for j in range(rep.shape[1]):
            weights_i = extract_weights(dataset, views[i], model, training_type)
            weights_j = extract_weights(dataset, views[j], model, training_type)
            for k in range(rep.shape[2]):
                top_bio_i = top_biomarkers(weights_i, Ks[k])
                top_bio_j = top_biomarkers(weights_j, Ks[k])
                rep[i,j,k] = sim(top_bio_i, top_bio_j)
    rep_ranks_ks = np.zeros(len(views) * len(Ks))
    
    for k in range(rep.shape[2]):
        rep_k = rep[:,:,k]
        rep_vec_k = np.sum(rep_k, axis=1)
        rep_ranks_ks[k*len(views):(k+1)*len(views)] = rep_vec_k.argsort()[::-1].argsort() #verified
    rep_s = np.sum(rep,axis=1)
    #rep_mean = np.mean(rep, axis=2)
    #rep_vec = np.sum(rep_mean, axis=1)
    rep_f = rep_s.flatten()
    rep_dict = {}
    rep_dict['strength_vector'] = rep_f
    rep_dict['rank_vector'] = rep_ranks_ks
    rep_dict['dataset'] = dataset
    rep_dict['views'] = views
    rep_dict['model'] = model
    rep_dict['training_type'] = training_type
    return rep_dict

def KL_symmetric(P,Q):
     epsilon = 0.00001
     P = P+epsilon
     Q = Q+epsilon
     divergence_pq = np.sum(P*np.log(P/Q))
     divergence_qp = np.sum(Q*np.log(Q/P))
     divergence = (divergence_qp+divergence_pq)/2 
     return divergence
 

def overall_rep_accumulated_cv_fixed(data_dict, training_type): # ongoing
    dataset = data_dict['dataset']
    views = data_dict['views']
    models = data_dict['models']
    rep_intersection_rank = np.zeros([len(models), len(models)])
    rep_intersection_weight = np.zeros([len(models), len(models)])
    rep_KL = np.zeros([len(models), len(models)])
    rep_L2 = np.zeros([len(models), len(models)])
    rep_corr = np.zeros([len(models), len(models)])
    for i in range(len(models)):
        rep_vect_i = GNN_specific_rep_accumulated_vect(dataset,views,training_type, models[i])
        rep_rank_i = rep_vect_i['rank_vector']
        rep_strength_i = rep_vect_i['strength_vector']
        for j in range(len(models)):
            rep_vect_j = GNN_specific_rep_accumulated_vect(dataset,views,training_type, models[j])
            rep_rank_j = rep_vect_j['rank_vector']
            rep_strength_j = rep_vect_j['strength_vector']
            if i==j:
                rep_intersection_rank[i,j] = 1.0
                rep_intersection_weight[i,j] = 1.0
            else:
                rep_intersection_rank[i,j] = sim_respective(rep_rank_i, rep_rank_j)
                rep_intersection_weight[i,j] = sim_respective_weighted(rep_rank_i, rep_rank_j, rep_strength_i, rep_strength_j)
            rep_KL[i,j] = KL_symmetric(rep_strength_i, rep_strength_j)
            rep_L2[i,j] = np.linalg.norm(rep_strength_i - rep_strength_j)
            corr_strength = np.corrcoef(rep_strength_i, rep_strength_j)
            rep_corr[i,j] = corr_strength[0,1] 
    rep_dict = {}
    rep_dict['intersection_rank'] = rep_intersection_rank
    rep_dict['intersection_weight'] = rep_intersection_weight
    rep_dict['correlation'] = rep_corr
    rep_dict['KL'] = rep_KL
    rep_dict['L2'] = rep_L2
    rep_dict['models'] = models
    rep_dict['dataset'] = dataset
    rep_dict['training_type'] = training_type
    return rep_dict
    
def overall_rep_accumulated(data_dict):
    models = data_dict['models']
    dataset = data_dict['dataset']
    training_types = data_dict['training_types']
    
    rep_intersection_rank = np.zeros([len(models), len(models), len(training_types)])
    rep_intersection_weight = np.zeros([len(models), len(models), len(training_types)])
    rep_KL = np.zeros([len(models), len(models), len(training_types)])
    rep_L2 = np.zeros([len(models), len(models), len(training_types)])
    rep_corr = np.zeros([len(models), len(models), len(training_types)])
    
    for i in range(len(training_types)):
        rep_dict = overall_rep_accumulated_cv_fixed(data_dict, training_types[i])
        rep_intersection_rank[:,:,i] = rep_dict['intersection_rank']
        rep_intersection_weight[:,:,i] = rep_dict['intersection_weight']
        rep_corr[:,:,i] = rep_dict['correlation']
        rep_KL[:,:,i] = rep_dict['KL']
        rep_L2[:,:,i] = rep_dict['L2']
        
    rep_intersection_rank_mean = np.mean(rep_intersection_rank, axis=2)
    rep_intersection_weight_mean = np.mean(rep_intersection_weight, axis=2)
    rep_corr_mean = np.mean(rep_corr, axis=2)
    rep_KL_mean = np.mean(rep_KL, axis=2)
    rep_L2_mean = np.mean(rep_L2, axis=2)
    rep_dict = {}
    rep_dict['rank intersection'] = rep_intersection_rank_mean
    rep_dict['weighted intersection'] = rep_intersection_weight_mean
    rep_dict['correlation'] = rep_corr_mean
    rep_dict['KL'] = rep_KL_mean
    rep_dict['L2'] = rep_L2_mean
    rep_dict['models'] = models
    rep_dict['dataset'] = dataset
    return rep_dict

def overall_rep_accumulated_plot(rep_dict, save_fig=False):
    models = rep_dict['models']
    keys = ['rank intersection', 'weighted intersection', 'correlation', 'KL', 'L2']
    a = 2  # number of rows
    b = 3  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure(figsize=(18,10))
    for k in keys:
        plt.subplot(a, b, c)
        plt.title(k)
        #plt.xlabel(k)
        df_k = pd.DataFrame(rep_dict[k], index = [i for i in models], columns = [i for i in models])
        sns.heatmap(df_k, annot=True ,vmin=df_k.to_numpy().min(), vmax=df_k.to_numpy().max())
        c = c + 1
    dataname_clean = rep_dict['dataset']
    dataname_clean.replace("_"," ")
    fig.suptitle('accumulated reproducibility matarices with accumulated GNN specific vectors '+dataname_clean)
    plt.show()
    if save_fig==True:
        plt.savefig("./imgs/Rep_accumulated"+ rep_dict['dataset'] + '_'  + ".png")

def manage_all_reps(data_dict):
    rep_avg = overall_avg_rep(data_dict)
    rep_corr = overall_corr_rep(data_dict)
    rep_accumulated = overall_rep_accumulated(data_dict)
    rep_dict = {}
    rep_dict['models'] = data_dict['models']
    rep_dict['dataset'] = data_dict['dataset']
    
    rep_dict['rank intersection (accumulated)'] = rep_accumulated['rank intersection']
    rep_dict['weighted intersection (accumulated)'] = rep_accumulated['weighted intersection']
    rep_dict['correlation (accumulated)'] = rep_accumulated['correlation']
    rep_dict['KL (accumulated)'] = rep_accumulated['KL']
    rep_dict['L2 (accumulated)'] = rep_accumulated['L2']
    rep_dict['rank correlation'] = rep_corr['rank_matrix']
    rep_dict['strength correlation'] = rep_corr['strength_matrix']
    rep_dict['views average'] = rep_avg['matrix']
    return rep_dict

def manage_all_reps_plot(rep_dict, save_fig=False):
    models = rep_dict['models']
    keys_rep = list(rep_dict.keys())
    keys_rep.remove('models')
    keys_rep.remove('dataset')
    #keys = ['rank intersection', 'weighted intersection', 'correlation', 'KL', 'L2']
    a = 2  # number of rows
    b = 4  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure(figsize=(25,10))
    for k in keys_rep:
        plt.subplot(a, b, c)
        plt.title(k)
        #plt.xlabel(k)
        df_k = pd.DataFrame(rep_dict[k], index = [i for i in models], columns = [i for i in models])
        sns.heatmap(df_k, annot=True ,vmin=df_k.to_numpy().min(), vmax=df_k.to_numpy().max())
        c = c + 1
    dataname_clean = rep_dict['dataset']
    dataname_clean.replace("_"," ")
    fig.suptitle('reproducibility matrices '+dataname_clean)
    plt.show()
    if save_fig==True:
        plt.savefig("./imgs/Rep_accumulated"+ rep_dict['dataset'] + '_'  + ".png")







'''# 1. avg
rep_avg = overall_avg_rep(data_dict)
overall_avg_rep_plot(rep_avg, save_fig = True)'''

'''# 2. corr
rep_corr = overall_corr_rep(data_dict)
overall_corr_rep_plot(rep_corr, corr_type='rank') 
overall_corr_rep_plot(rep_corr, corr_type='strength', save_fig = True) '''

# 3. accuulated Ks
#aa = overall_rep_accumulated(data_dict)
#overall_rep_accumulated_plot(aa)

data_dict={}
data_dict['dataset'] = 'Demo' # 'LH_ADLMCI'
data_dict['views'] = [0, 1, 2, 3] #number of views 
data_dict['models'] = ['diffpool', 'gat', 'gcn', 'gunet', 'sag']
data_dict['training_types'] = ['3Fold'] #'Few_Shot'

rep_dict = manage_all_reps(data_dict)
name = data_dict['dataset'] + '_cv.pickle'
with open(name, 'wb') as f:
    pickle.dump(rep_dict, f)

data_dict={}
data_dict['dataset'] = 'Demo' # 'LH_ADLMCI'
data_dict['views'] = [0, 1, 2, 3] #number of views 
data_dict['models'] = ['diffpool', 'gat', 'gcn', 'gunet', 'sag']
data_dict['training_types'] = ['Few_Shot'] #'Few_Shot'

rep_dict = manage_all_reps(data_dict)
name = data_dict['dataset'] + '_fs.pickle'
with open(name, 'wb') as f:
    pickle.dump(rep_dict, f)





