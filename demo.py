#!/usr/bin/python
"""
Copyright 2020 Furkan Tornaci, Istanbul Technical University.
All rights reserved.
"""

from split_cv import transform_Data
from split_Few_shot import transform_Data_FewShot
from Analysis import  new_folder, Rep_histograms, Models_trained, Rep_heatmap

import argparse
import os
import numpy as np
import torch
import random
import main_diffpool
import main_gat
import main_gcn
import main_gunet
import main_sag

def train_main_model(dataset,model,view, cv_number):
    
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network. 
    
    Description
    ----------
    This method trains selected GNN model with 5-Fold Cross Validation.
    
    """
    name = str(cv_number)+"Fold"
    #name = "5Fold"
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)    
    model_name = "MainModel_"+name+"_"+dataset+ "_" + model
    new_folder(model)
    if model=='diffpool':
        main_diffpool.test_scores(dataset, view, model_name, cv_number)
    elif model=='gcn':
        main_gcn.test_scores(dataset, view, model_name, cv_number)
    elif model=='gat':
        main_gat.test_scores(dataset, view, model_name, cv_number)
    elif model == "gunet":
        transform_Data(cv_number, dataset)
        main_gunet.cv_benchmark(dataset, view, cv_number)
    elif model == "sag":
        transform_Data(cv_number, dataset)
        main_sag.cv_benchmark(dataset, view, cv_number)

def two_shot_train(dataset, model, view, num_shots):
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network.
    
    Description
    ----------
    This method trains selected GNN model with Two shot learning.
    
    """
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    transform_Data_FewShot(dataset)
    new_folder(model)
    if model == "gunet":
        main_gunet.two_shot_trainer(dataset, view, num_shots)
    elif model == "gcn":
        main_gcn.two_shot_trainer(dataset, view, num_shots)
    elif model == "gat":
        main_gat.two_shot_trainer(dataset, view, num_shots)
    elif model == "diffpool":
        main_diffpool.two_shot_trainer(dataset, view, num_shots)
    elif model == "sag":
        main_sag.two_shot_trainer(dataset, view, num_shots)

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'results'])
    #parser.add_argument('--v', type=str, default=0, help='index of cortical morphological network.')
    parser.add_argument('--cv_number', type=str, default=3, help='number of cross validations.')
    parser.add_argument('--num_shots', type=str, default=5, help='number of runs for the FS learning.')
    #parser.add_argument('--data', type=str, default='Demo', choices = [ f.path[5:] for f in os.scandir("data") if f.is_dir() ], help='Name of dataset')
    args = parser.parse_args()
    #view = args.v
    #dataset = args.data
    num_shots = args.num_shots
    cv_number = args.cv_number
    
    if args.mode == 'train':
        '''
        Training GNN Models with datasets of data directory.
        '''
        #datasets_asdnc = ['Demo']
        datasets_asdnc = ['RH_ASDNC','LH_ASDNC']
        #datasets_adlmci = ['RH_ADLMCI','LH_ADLMCI']
        
        
        views = [0, 1, 2, 3]
        for dataset_i in datasets_asdnc:
            for view_i in views:
                models = ["sag"]
                #models = ["gunet"]
                for model in models:
                    two_shot_train(dataset_i, model, view_i, num_shots)
                    train_main_model(dataset_i, model, view_i, 3)
        
            print("All GNN architectures are trained with dataset: "+dataset_i)
          
        
    elif args.mode == 'results':
        '''
        if Models_trained(dataset, view):
            print("Models are not trained with"+dataset+" dataset view:"+str(view))
        else:
            Rep_histograms(dataset, view)
            Rep_heatmap(dataset, view)
            print("Reproducibility Histogram of dataset "+dataset+" is saved into results file.")
        '''


