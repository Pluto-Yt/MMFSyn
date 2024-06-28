import argparse
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gcn3 import GCNNet
from utils import *
from utils import collate_fn
from sklearn.model_selection import KFold

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    #print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_druga = []
    total_drugb = []
    total_cell_line = []
    total_tissue = []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            total_druga.extend(data.drug_A)
            total_drugb.extend(data.drug_B)
            total_cell_line.extend(data.cee_line)
            total_tissue.extend(data.tissue)
    
    return total_druga,total_drugb,total_cell_line,total_tissue,total_labels.numpy().flatten(),total_preds.numpy().flatten()

loss_fn = nn.MSELoss()
LOG_INTERVAL = 200

def main(args):
  dataset = args.dataset
  modeling = [GCNNet]
  model_st = modeling[0].__name__

  cuda_name = "cuda:0"
  print('cuda_name:', cuda_name)

  TRAIN_BATCH_SIZE = args.batch_size
  TEST_BATCH_SIZE = args.batch_size
  LR = args.lr
  
  NUM_EPOCHS = args.epoch

  print('Learning rate: ', LR)
  print('Epochs: ', NUM_EPOCHS)

  # Main program: iterate over different datasets
  print('\nrunning on ', model_st + '_' + dataset )
  processed_data_file_train = '/home/yt/code/MMFSyn/predictor/data/processed/' + dataset + 'synergy.pt'
  #processed_data_file_test = '/home/yt/code/MMFSyn/predictor/data/processed/' + dataset + 'test0.pt'
  if ((not os.path.isfile(processed_data_file_train))): # or (not os.path.isfile(processed_data_file_test))
     print('please run create_data.py to prepare data in pytorch format!')
  else:
    train_data = TestbedDataset(root='/home/yt/code/MMFSyn/predictor/data', dataset=dataset+'synergy')
    #test_data = TestbedDataset(root='/home/yt/code/MMFSyn/predictor/data', dataset=dataset+'test0')
        

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    result_file_name = 'synergy-11.22-3D-2Dgmlp-morgan-HGCN-BiLSTM' + model_st + '_' + '.csv'

    ii = 1
    MSEfinal = []
    rMSEfinal = []
    Pfinal = []
    Spearmanfinal = []
    Cifinal = []
    rm2final = []


    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(train_data):
      best_mse = 1000
      best_ci = 0
      best_epoch = -1

      print("第{}折交叉验证".format(ii))
      train_fold = torch.utils.data.dataset.Subset(train_data, train_index)
      val_fold = torch.utils.data.dataset.Subset(train_data, val_index)    
 
      # 打包成DataLoader类型 用于 训练
      train_loader = DataLoader(dataset=train_fold, batch_size=256, shuffle=True,num_workers = 8)
      test_loader = DataLoader(dataset=val_fold, batch_size=256, shuffle=True,num_workers = 8)
      train_size = len(train_loader)
      test_size = len(test_loader)
      model = modeling[0](k1=1,k2=2,k3=3,embed_dim=78,num_layer=1,device=device).to(device)
      optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
      for epoch in range(NUM_EPOCHS):
        #hidden,cell = model.init_hidden(batch_size=TRAIN_BATCH_SIZE)
        train(model, device, train_loader, optimizer, epoch+1) #,hidden,cell
        total_druga,total_drugb,total_cell_line,total_tissue,G,P = predicting(model, device, test_loader) #,hidden,cell
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),get_rm2(G.reshape(G.shape[0],-1),P.reshape(P.shape[0],-1))]
        if ret[1]<best_mse:
          best_epoch = epoch+1
          best_rmse = ret[0]  
          best_mse = ret[1]
          best_pearson = ret[2] 
          best_spearman = ret[3]
          best_ci = ret[4]
          best_rm2 = ret[5]
          print('rmse improved at epoch ', best_epoch, '; best:', best_rmse,best_mse,best_pearson,best_spearman,best_ci,best_rm2)

        else:
          # print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
          print(ret[1],'No improvement since epoch ', best_epoch, '; best:', best_rmse,best_mse,best_pearson,best_spearman,best_ci,best_rm2)

      #del model,G,P
      MSEfinal.append(best_mse)
      rMSEfinal.append(best_rmse)
      Pfinal.append(best_pearson)
      Spearmanfinal.append(best_spearman)
      Cifinal.append(best_ci)
      rm2final.append(best_rm2)

      ii += 1
    print("rmse为:",rMSEfinal)
    print("最终rmse为:",sum(rMSEfinal)/len(rMSEfinal))

    print("mse为:",MSEfinal)
    print("最终mse为:",sum(MSEfinal)/len(MSEfinal))

    print("Pearson为:",Pfinal)
    print("最终Pearson为:",sum(Pfinal)/len(Pfinal))

    print("Spearman为:",Spearmanfinal)
    print("最终Spearman为:",sum(Spearmanfinal)/len(Spearmanfinal))

    print("CI为:",Cifinal)
    print("最终CI为:",sum(Cifinal)/len(Cifinal))

    print("rm2为:",rm2final)
    print("最终rm2为:",sum(rm2final)/len(rm2final))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run DeepGLSTM")

  parser.add_argument("--dataset",type=str,default='',
                      help="Dataset Name (davis,kiba,DTC,Metz,ToxCast,Stitch)")

  parser.add_argument("--epoch",
                      type = int,
                      default = 400,
                      help="Number of training epochs. Default is 1000."
                      ) 
  
  parser.add_argument("--lr",
                      type=float,
                      default = 0.00001,
                      help="learning rate",
                      )
  
  parser.add_argument("--batch_size",type=int,
                      default = 512,
                      help = "Number of drug-tareget per batch. Default is 128 for davis.") # batch 128 for Davis
  
  parser.add_argument("--save_file",type=str,
                      default='',
                      help="Where to save the trained model. For example davis.model") 


  args = parser.parse_args()
  print(args)
  main(args)
