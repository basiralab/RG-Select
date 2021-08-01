# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:33:19 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:15:56 2021

@author: user
"""

import pickle
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import torch

plt.rcParams['font.family'] = "serif"

def gnn_scores(rep_dict, tt, save_fig=False):
    models = rep_dict['models']
    
    keys_rep_new = ['views average','rank correlation']
    # 'views average','rank correlation',
    # 'weighted intersection (accumulated)', 'correlation (accumulated)',
    # 'strength correlation','rank intersection (accumulated)'
    fig = plt.figure(figsize=(25,10))  #(25,10)
    ax = fig.add_gridspec(16,37)
    
    # sub 1
    ax_1 = fig.add_subplot(ax[1:7, 0:7])
    rep_1 = rep_dict[keys_rep_new[0]]
    df_1 = pd.DataFrame(rep_1, index = [i for i in models], columns = [i for i in models])
    sns_1 = sns.heatmap(df_1, ax=ax_1, annot=True ,vmin=df_1.to_numpy().min(), 
                        vmax=df_1.to_numpy().max(), cmap='Blues') #GnBu, Blues, PuBu, vlag, ocean
    #sns_1.set_xticklabels(sns_1.get_xticklabels(),fontsize=35/np.sqrt(len(rep_1)))
    sns_1.set_yticklabels(sns_1.get_yticklabels(), va='center')
    plt.title(keys_rep_new[0])
    
    # sub 1 vector
    ax_1_v = fig.add_subplot(ax[1:7, 6:10])
    rep_1_v = (np.sum(rep_1,axis=0)-1)/(len(models)-1)
    df_1_v = pd.DataFrame(rep_1_v, index = [i for i in models])
    sns_1_v = sns.heatmap(df_1_v, ax=ax_1_v,annot=True ,vmin=df_1_v.to_numpy().min(), 
                        vmax=df_1_v.to_numpy().max(), cmap='Blues', xticklabels=False, square=True, cbar_kws={'shrink':1}) #GnBu, Blues, PuBu, vlag, ocean
    sns_1_v.set_yticklabels(sns_1_v.get_yticklabels(), va='center')
    plt.title('scores')
    
    # sub 2
    ax_2 = fig.add_subplot(ax[8:14, 0:7])
    rep_2 = rep_dict[keys_rep_new[1]]
    df_2 = pd.DataFrame(rep_2, index = [i for i in models], columns = [i for i in models])
    sns_2 = sns.heatmap(df_2, ax=ax_2, annot=True ,vmin=df_2.to_numpy().min(), 
                        vmax=df_2.to_numpy().max(), cmap='Purples') #GnBu, Blues, PuBu, vlag, ocean
    #sns_2.set_xticklabels(sns_2.get_xticklabels(),fontsize=35/np.sqrt(len(rep_2)))
    sns_2.set_yticklabels(sns_2.get_yticklabels(), va='center')
    plt.title(keys_rep_new[1])
    
    # sub 2 vector
    ax_2_v = fig.add_subplot(ax[8:14, 6:10])
    rep_2_v = (np.sum(rep_2,axis=0)-1)/(len(models)-1)
    df_2_v = pd.DataFrame(rep_2_v, index = [i for i in models])
    sns_2_v = sns.heatmap(df_2_v, ax=ax_2_v,annot=True ,vmin=df_2_v.to_numpy().min(), 
                        vmax=df_2_v.to_numpy().max(), cmap='Purples', xticklabels=False, square=True, cbar_kws={'shrink':1}) #GnBu, Blues, PuBu, vlag, ocean
    sns_2_v.set_yticklabels(sns_2_v.get_yticklabels(), va='center')
    plt.title('scores')
     
    # sub overall matrix
    rep_sum = np.zeros((len(models),len(models))) 
    for k in keys_rep_new[:3]:
        rep_sum = rep_sum + rep_dict[k]
    ax_sum = fig.add_subplot(ax[3:12, 15:26])
    df_sum = pd.DataFrame(rep_sum, index = [i for i in models], columns = [i for i in models])
    sns_m = sns.heatmap(df_sum, ax=ax_sum, annot=True ,vmin=df_sum.to_numpy().min(), 
                        annot_kws={"size":14},vmax=df_sum.to_numpy().max(), cmap='PuBu') #GnBu, Blues, PuBu, vlag, ocean
    sns_m.set_xticklabels(sns_m.get_xticklabels(),fontsize=14)
    sns_m.set_yticklabels(sns_m.get_yticklabels(), va='center',fontsize=14)
    plt.title('overall reproducibility', fontsize=18)
    
    # sub overall vector
    ax_vec = fig.add_subplot(ax[3:12, 16:])
    #np_vec = (np.sum(rep_sum,axis=0)-4)/(len(models)-1)
    np_vec = (np.sum(rep_sum,axis=0)-3)/(len(models)-1)
    df_vec = pd.DataFrame(np_vec, index = [i for i in models])
    sns_v = sns.heatmap(df_vec, ax=ax_vec,annot=True ,vmin=df_vec.to_numpy().min(), 
                        annot_kws={"size":14},vmax=df_vec.to_numpy().max(), cmap='PuBu', xticklabels=False, square=True, cbar_kws={'shrink':1}) #GnBu, Blues, PuBu, vlag, ocean
    sns_v.set_yticklabels(sns_v.get_yticklabels(), va='center',fontsize=14)
    plt.title('scores', fontsize=18)
    
    # title
    dataname = rep_dict['dataset']
    fig.suptitle('Overall reproducibility matrix for '+dataname+' dataset \n with ' + tt +' training', fontsize=24)
    return rep_sum

datasets= ['Demo']
#datasets= ['RH_ADLMCI']
for dataset_i in datasets:
    rep_data = dataset_i + '_cv.pickle'
    with open(rep_data,'rb') as f:
        rep_dict = pickle.load(f)
    gnn_scores(rep_dict, 'CV', save_fig=True)
    
for dataset_i in datasets:
    rep_data = dataset_i + '_fs.pickle'
    with open(rep_data,'rb') as f:
        rep_dict = pickle.load(f)
    gnn_scores(rep_dict, 'FS', save_fig=True)