
from models_gunet import GNet
from gunet_trainer import Trainer
from torch.autograd import Variable
from graph_sampler import GraphSampler

import argparse
import random
import time
import torch
import numpy as np
import pickle
import sklearn.metrics as metrics
import Analysis
import os 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_args(num_shots=2, cv_number=5):
    parser = argparse.ArgumentParser(description='Args for graph predition')
    
    
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--v', type=str, default=1)
    parser.add_argument('--data', type=str, default='Sample_dataset', choices = [ f.path[5:] for f in os.scandir("data") if f.is_dir() ])
    
    
    
    parser.add_argument('-num_classes', type=int, default=2, help='seed')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='DD', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=1, help='epochs') #35
    parser.add_argument('--num_shots', type=int, default=num_shots, help='number of shots') #100
    parser.add_argument('-batch', type=int, default=1, help='batch size')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('-deg_as_tag', type=int, default=1, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=48, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.9, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.9, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7])
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    parser.add_argument('--threshold', dest='threshold', default='mean',
            help='threshold the graph adjacency matrix. Possible values: no_threshold, median, mean')
    args, _ = parser.parse_known_args()
    return args

def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def app_run(args, G_data, fold_idx):
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    trainer = Trainer(args, net, G_data)
    trainer.train()

def evaluate(dataset, model, args, model_name):
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
    model.eval()
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
    
        #batch_num_nodes=np.array([adj.shape[1]])
        
        h0 = np.identity(adj.shape[1])
        h0 = Variable(torch.from_numpy(h0).float(), requires_grad=False).cpu()
        adj = torch.squeeze(adj)
        
        ypred = model([adj] ,[h0])

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    simple_r = {'labels':labels,'preds':preds}

    with open("./gunet/Labels_and_preds/"+model_name+".pickle", 'wb') as f:
      pickle.dump(simple_r, f)


    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    
    print("Test accuracy:", result['acc'])
    return result['acc']

def train(args, train_dataset, val_dataset, model, model_name):
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
    params = list(model.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        print("Epoch ",epoch)
        
        model.train()
        total_time = 0
        avg_loss = 0.0
        
        preds = []
        labels = []
        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()
            
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            #adj_id = Variable(data['id'].int()).to(device)
            
            h0 = np.identity(adj.shape[1])
            h0 = Variable(torch.from_numpy(h0).float(), requires_grad=False).cpu()
            adj = torch.squeeze(adj)

            
            
            ypred = model([adj] ,[h0])
            
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            
            loss = model.loss_metric(ypred, label)
            
            model.zero_grad()
            
            loss.backward()
        
            optimizer.step()
            
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
            
        if epoch==args.num_epochs-1:
              Analysis.is_trained = True
              
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        
        test_acc = evaluate(val_dataset, model, args, model_name)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        train_loss.append(avg_loss)


    path = './gunet/weights/W_'+model_name+'.pickle'
    
    if os.path.exists(path):
        os.remove(path)
        
    os.rename('Gunet_W.pickle', path)

    los_p = {'loss':train_loss}
    with open("./gunet/training_loss/Training_loss_"+model_name+".pickle", 'wb') as f:
      pickle.dump(los_p, f)
    torch.save(model,"./gunet/models/Gunet_"+model_name+".pt")
    return test_acc

def create_data_loaders(train, validation):
    print('Num training graphs: ', len(train), 
          '; Num test graphs: ', len(validation))
    
    # minibatch
    dataset_sampler = GraphSampler(train)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size = 1,  
            shuffle = False)  

    dataset_sampler = GraphSampler(validation)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size = 1,  
            shuffle = False) 
    return train_dataset_loader, val_dataset_loader

def cv_benchmark(dataset, view, cv_number):
    
    cv = cv_number
    model = "gunet"
    name = str(cv)+"Fold"
    model_name = "MainModel_"+name+"_"+dataset+ "_" + model
    
    args = get_args()
    print(args)
    set_random(args.seed)
    start = time.time()
    
    if not os.path.exists('Folds'+str(cv)):
        os.makedirs('Folds'+str(cv))
    
    for i in range(cv):
        print("CV : ",i)
        with open('./Folds_views'+str(cv)+'/'+dataset+'_view_'+str(view)+'_folds_'+ str(cv) +'_fold_'+str(i)+'_train','rb') as f:
            train_set = pickle.load(f)
        with open('./Folds_views'+str(cv)+'/'+dataset+'_view_'+str(view)+'_folds_'+ str(cv) +'_fold_'+str(i)+'_test','rb') as f:
            test_set = pickle.load(f)
        feat_dim = train_set[0]['adj'].shape[0]
        # dataloaders
        train_loader, val_loader = create_data_loaders(train_set, test_set)
        #test_loader = DataLoader(test_set,batch_size=1, shuffle=True)
        # net
        net = GNet(feat_dim, args.num_classes, args)
        test_acc = train(args, train_loader, val_loader, net, model_name+"_CV_"+str(i)+"_view_"+str(view))
        print("Test accuracy:"+str(test_acc))
        print('load data using ------>', time.time()-start)
        

def two_shot_trainer(dataset, view, num_shots):
    args = get_args(num_shots)
    print(args)
    set_random(args.seed)
    start = time.time()
    
    for i in range(args.num_shots):
        model = "gunet"
        model_name = "Few_Shot_"+dataset+"_"+model + str(i)
        print("Shot : ",i)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_train','rb') as f:
            train_set = pickle.load(f)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_test','rb') as f:
            test_set = pickle.load(f)
        feat_dim = train_set[0]['adj'].shape[0]
        # dataloaders
        train_loader, val_loader = create_data_loaders(train_set, test_set)
        #test_loader = DataLoader(test_set,batch_size=1, shuffle=True)
        # net
        net = GNet(feat_dim, args.num_classes, args)
        test_acc = train(args, train_loader, val_loader, net, model_name+"_view_"+str(view))
        print("Test accuracy:"+str(test_acc))
        print('load data using ------>', time.time()-start)
        

