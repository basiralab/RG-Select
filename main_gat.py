# -*- coding: utf-8 -*-
from sklearn import preprocessing
from torch.autograd import Variable
from models_gat import GAT

import os
import torch
import numpy as np
import argparse
import pickle
import sklearn.metrics as metrics
import cross_val
import time
import random


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(dataset, model_GAT, args, threshold_value, model_name):
    """
    Parameters
    ----------
    dataset : dataloader (dataloader for the validation/test dataset).
    model_GCN : nn model (GAT model).
    args : arguments
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the evaluation of the model on test/validation dataset
    
    Returns
    -------
    test accuracy.
    """
    model_GAT.eval()
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        
        adj = torch.squeeze(adj)
        
        features = np.identity(adj.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
        if args.threshold in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
        
        ypred = model_GAT(features, adj)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    simple_r = {'labels':labels,'preds':preds}

    with open("./gat/Labels_and_preds/"+model_name+".pickle", 'wb') as f:
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

def train(args, train_dataset, val_dataset, model_GAT, threshold_value, model_name):
    """
    Parameters
    ----------
    args : arguments
    train_dataset : dataloader (dataloader for the validation/test dataset).
    val_dataset : dataloader (dataloader for the validation/test dataset).
    model_GAT : nn model (GAT model).
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the training of the model on train dataset and calls evaluate() method for evaluation.
    
    Returns
    -------
    test accuracy.
    """
    params = list(model_GAT.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    test_accs = []
    train_loss=[]
    val_acc=[]    

    for epoch in range(args.num_epochs):
        print("Epoch ",epoch)
        
        print("Size of Training Set:" + str(len(train_dataset)))
        print("Size of Validation Set:" + str(len(val_dataset)))    
        model_GAT.train()
        total_time = 0
        avg_loss = 0.0
        
        preds = []
        labels = []
        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()
            
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            #adj_id = Variable(data['id'].int()).to(device)
            
            adj = torch.squeeze(adj)

            features = np.identity(adj.shape[0])
            features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
            
            if args.threshold in ["median", "mean"]:
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
            
            
            ypred = model_GAT(features, adj)
            
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            loss = model_GAT.loss(ypred, label)
            model_GAT.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model_DIFFPOOL.parameters(), args.clip)
            optimizer.step()
            
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        
        if epoch == args.num_epochs-1:
            model_GAT.is_trained = True
        
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        test_acc = evaluate(val_dataset, model_GAT, args, threshold_value, model_name)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        test_accs.append(test_acc)
        train_loss.append(avg_loss)
        val_acc.append(test_acc)
        

    path = './gat/weights/W_'+model_name+'.pickle'
    
    if os.path.exists(path):
        os.remove(path)
        
    os.rename('GAT_W.pickle',path)
  
    los_p = {'loss':train_loss}
    with open("./gat/training_loss/Training_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    torch.save(model_GAT,"./gat/models/GAT_"+model_name+".pt")

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
    #Load graphs and labels
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
    arguments definition method
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
    parser.add_argument('--threshold', dest='threshold', default='mean',
                        help='threshold the graph adjacency matrix. Possible values: no_threshold, median, mean')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                        help='Number of label classes')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, 
                        help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, 
                        help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.8, 
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, 
                        help='Alpha for the leaky_relu.')
    
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

        print("CV : ",i)
        model_GAT = GAT(nfeat=num_nodes, 
                nhid=args.hidden, 
                nclass=args.num_classes, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
        
        test_acc = train(args, train_dataset, val_dataset, model_GAT, threshold_value, model_name+"_CV_"+str(i)+"_view_"+str(args.view))
        test_accs.append(test_acc)
    return test_accs

def test_scores(dataset, view, model_name, cv_number):
    
    args = arg_parse(dataset, view, cv_number=cv_number)
    print("Main : ",args)
    test_accs = benchmark_task(args, model_name)
    print("test accuracies ",test_accs)
    return test_accs
    

def two_shot_trainer(dataset, view, num_shots):
    args = arg_parse(dataset, view, num_shots=num_shots)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  
    start = time.time()
    
    for i in range(args.num_shots):
        model = "gat"
        model_name = "Few_Shot_"+dataset+"_"+model + str(i)
        print("Shot : ",i)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_train','rb') as f:
            train_set = pickle.load(f)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_test','rb') as f:
            test_set = pickle.load(f)
        
        num_nodes = train_set[0]['adj'].shape[0]
        
        model_GAT = GAT(nfeat=num_nodes, 
                nhid=args.hidden, 
                nclass=args.num_classes, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
        
        train_dataset, val_dataset, threshold_value = cross_val.two_shot_loader(train_set, test_set, args)
        
        test_acc = train(args, train_dataset, val_dataset, model_GAT, threshold_value, model_name+"_view_"+str(view))
        
        print("Test accuracy:"+str(test_acc))
        print('load data using ------>', time.time()-start)