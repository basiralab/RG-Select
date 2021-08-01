# -*- coding: utf-8 -*-

from sklearn import preprocessing
from torch.autograd import Variable

import os
import torch
import numpy as np
import argparse
import pickle
import sklearn.metrics as metrics

import cross_val
import models_diffpool as model_diffpool
import Analysis

import time
import random


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate(dataset, model_DIFFPOOL, args, threshold_value, model_name):
    """
    Parameters
    ----------
    dataset : dataloader (dataloader for the validation/test dataset).
    model_GCN : nn model (diffpool model).
    args : arguments
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the evaluation of the model on test/validation dataset
    
    Returns
    -------
    test accuracy.
    """
    model_DIFFPOOL.eval()
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
    
        batch_num_nodes=np.array([adj.shape[1]])
        
        h0 = np.identity(adj.shape[1])
        h0 = Variable(torch.from_numpy(h0).float(), requires_grad=False).cpu()
        h0 = torch.unsqueeze(h0, 0)
        
        assign_input = np.identity(adj.shape[1])
        assign_input = Variable(torch.from_numpy(assign_input).float(), requires_grad=False).cpu()
        assign_input = torch.unsqueeze(assign_input, 0)
        
        if args.threshold in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))

        ypred = model_DIFFPOOL(h0, adj, batch_num_nodes, assign_x=assign_input)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    simple_r = {'labels':labels,'preds':preds}

    with open("./diffpool/Labels_and_preds/"+model_name+".pickle", 'wb') as f:
      pickle.dump(simple_r, f)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    if args.evaluation_method == 'model assessment':
        name = 'Test'
    if args.evaluation_method == 'model selection':
        name = 'Validation'
    print(name, " accuracy:", result['acc'])
    return result['acc']

def minmax_sc(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

def train(args, train_dataset, val_dataset, model_DIFFPOOL, threshold_value, model_name):
    """
    Parameters
    ----------
    args : arguments
    train_dataset : dataloader (dataloader for the validation/test dataset).
    val_dataset : dataloader (dataloader for the validation/test dataset).
    model_DIFFPOOL : nn model (diffpool model).
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the training of the model on train dataset and calls evaluate() method for evaluation.
    
    Returns
    -------
    test accuracy.
    """
    
    train_loss=[]
    val_acc=[]

    params = list(model_DIFFPOOL.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        print("Epoch ",epoch)
        
        print("Size of Training Set:" + str(len(train_dataset)))
        print("Size of Validation Set:" + str(len(val_dataset)))
        model_DIFFPOOL.train()
        total_time = 0
        avg_loss = 0.0
        
        preds = []
        labels = []
        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()
            
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            #adj_id = Variable(data['id'].int()).to(device)
            
            batch_num_nodes=np.array([adj.shape[1]])
            
            h0 = np.identity(adj.shape[1])
            h0 = Variable(torch.from_numpy(h0).float(), requires_grad=False).cpu()
            h0 = torch.unsqueeze(h0, 0)

            assign_input = np.identity(adj.shape[1])
            assign_input = Variable(torch.from_numpy(assign_input).float(), requires_grad=False).cpu()
            assign_input = torch.unsqueeze(assign_input, 0)
            if args.threshold in ["median", "mean"]:
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
            
            

            ypred = model_DIFFPOOL(h0, adj , batch_num_nodes, assign_x=assign_input)
            
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            
            
            loss = model_DIFFPOOL.loss(ypred, label)
            
            model_DIFFPOOL.zero_grad()
            
            loss.backward()
            #nn.utils.clip_grad_norm_(model_DIFFPOOL.parameters(), args.clip)
            optimizer.step()
            
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
            
        if epoch==args.num_epochs-1:
              Analysis.is_trained = True
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        #result_train = evaluate(train_dataset, model_GTN, model_DIFFPOOL, args)
        test_acc = evaluate(val_dataset, model_DIFFPOOL, args, threshold_value, model_name)
        val_acc.append(test_acc)
        train_loss.append(avg_loss)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        #tracked_Dicts.append(tracked_Dict)

    ## Rename weight file
    
    path = './diffpool/weights/W_'+model_name+'.pickle'
    
    if os.path.exists(path):
        os.remove(path)
    
    os.rename('Diffpool_W.pickle',path)
    
    los_p = {'loss':train_loss}
    with open("./diffpool/training_loss/Training_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    torch.save(model_DIFFPOOL,"./diffpool/models/Diffpool_"+model_name+".pt")
    return test_acc

def load_data(args):
    """
    Parameters
    ----------
    args : arguments
    Description
    ----------
    This methods loads the adjacency matrices representing the args.view -th view in dataset
    
    Returns
    -------
    List of dictionaries{adj, label, id}
    """
    with open('data/'+args.dataset+'/'+args.dataset+'_edges','rb') as f:
        multigraphs = pickle.load(f)        
    with open('data/'+args.dataset+'/'+args.dataset+'_labels','rb') as f:
        labels = pickle.load(f)
    adjacencies = [multigraphs[i][:,:,args.view] for i in range(len(multigraphs))]
    #Normalize inputs
    if args.NormalizeInputGraphs==True:
        for subject in range(len(adjacencies)):
            adjacencies[subject] = minmax_sc(adjacencies[subject])
    
    #Create List of Dictionaries
    G_list=[]
    for i in range(len(labels)):
        G_element = {"adj":   adjacencies[i],"label": labels[i],"id":  i,}
        G_list.append(G_element)
    return G_list

def arg_parse(dataset, view, num_shots=2, cv_number=5):
    """
    args definition method
    """
    parser = argparse.ArgumentParser(description='Graph Classification')
    
    
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--v', type=str, default=1)
    parser.add_argument('--data', type=str, default='Sample_dataset', choices = [ f.path[5:] for f in os.scandir("data") if f.is_dir() ])
    
    

    
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='Dataset')
    parser.add_argument('--view', type=int, default=view,
                        help = 'view index in the dataset')
    parser.add_argument('--num_epochs', type=int, default=1, #50
                        help='Training Epochs')
    parser.add_argument('--num_shots', type=int, default=num_shots, #100
                        help='number of shots')
    parser.add_argument('--cv_number', type=int, default=cv_number,
                        help='number of validation folds.')
    parser.add_argument('--NormalizeInputGraphs', default=False, action='store_true',
                        help='Normalize Input adjacency matrices of graphs')  
    parser.add_argument('--evaluation_method', type=str, default='model assessment',
                        help='evaluation method, possible values : model selection, model assessment')
    ##################
    parser.add_argument('--lr', type=float, default = 0.0001,
                    help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default = 0.00001,
                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--threshold', dest='threshold', default='median',
            help='threshold the graph adjacency matrix. Possible values: no_threshold, median, mean')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int, default=512,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--assign-ratio', dest='assign_ratio', type=float, default=0.1,
                        help='ratio of number of nodes in consecutive layers')
    parser.add_argument('--num-pool', dest='num_pool', type=int, default=1,
                        help='number of pooling layers')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--clip', dest='clip', type=float, default=2.0,
            help='Gradient clipping.')
    
    return parser.parse_args()

def benchmark_task(args, model_name):
    """
    Parameters
    ----------
    args : Arguments
    Description
    ----------
    Initiates the model and performs train/test or train/validation splits and calls train() to execute training and evaluation.
    Returns
    -------
    test_accs : test accuracies (list)

    """
    G_list = load_data(args)
    num_nodes = G_list[0]['adj'].shape[0]
    test_accs = []
    folds = cross_val.stratify_splits(G_list,args)
    [random.shuffle(folds[i]) for i in range(len(folds))]
    for i in range(args.cv_number):
        train_set, validation_set, test_set = cross_val.datasets_splits(folds, args, i)
        if args.evaluation_method =='model selection':
            train_dataset, val_dataset, threshold_value = cross_val.model_selection_split(train_set, validation_set, args)
        if args.evaluation_method =='model assessment':
            train_dataset, val_dataset, threshold_value = cross_val.model_assessment_split(train_set, validation_set, test_set, args)
        assign_input = num_nodes
        input_dim = num_nodes
        print("CV : ",i)
        model_DIFFPOOL = model_diffpool.SoftPoolingGcnEncoder(
                    num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input).cpu()
        
        test_acc = train(args, train_dataset, val_dataset, model_DIFFPOOL, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(args.view))
        test_accs.append(test_acc)
    return test_accs

def test_scores(dataset, view, model_name, cv_number):
    
    args = arg_parse(dataset, view, cv_number=cv_number)
    print("Main : ",args)
    test_accs = benchmark_task(args, model_name)
    print("test accuracies ",test_accs)
    return test_accs
    
def two_shot_trainer(dataset, view, num_shots):
    args = arg_parse(dataset, view, num_shots)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  
    start = time.time()
    
    for i in range(args.num_shots):
        model = "diffpool"
        model_name = "Few_Shot_"+dataset+"_"+model + str(i)
        print("Shot : ",i)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_train','rb') as f:
            train_set = pickle.load(f)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_test','rb') as f:
            test_set = pickle.load(f)
        
        num_nodes = train_set[0]['adj'].shape[0]
        
        assign_input = num_nodes
        input_dim = num_nodes
        
        model_DIFFPOOL = model_diffpool.SoftPoolingGcnEncoder(
                    num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input).cpu()
        
        train_dataset, val_dataset, threshold_value = cross_val.two_shot_loader(train_set, test_set, args)
        
        test_acc = train(args, train_dataset, val_dataset, model_DIFFPOOL, threshold_value, model_name+"_view_"+str(view))

        print("Test accuracy:"+str(test_acc))
        print('load data using ------>', time.time()-start)

    