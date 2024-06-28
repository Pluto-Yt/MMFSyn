import argparse
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import json,pickle
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    #mol = Chem.AddHs(mol)

    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    c_size, features, edge_index= torch.Tensor(c_size), torch.Tensor(features), torch.LongTensor(edge_index)
    return c_size, features, edge_index


    

def main(args):
  #dataset = args.dataset

#   compound_iso_smiles = []
#   opts = ['train','test']
#   for opt in opts:
#     df = pd.read_csv('/home/yt/code/MMFSyn/predictor/data/' + opt + '.csv') #dataset + '_' 
#     compound_iso_smiles += list( df['drug_a'] )
#   compound_iso_smiles = set(compound_iso_smiles)

  df = pd.read_csv('/home/yt/code/MMFSyn/drug/data/smiles.csv')
  compound_iso_smiles = list(df['smile'])
  smile_graph = {}
  for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

  # convert to torch geometric data
  processed_data_file_train = '/home/yt/code/MMFSyn/predictor/data/processed/' + 'synergy.pt' #dataset + 
  if ((not os.path.isfile(processed_data_file_train))): # or (not os.path.isfile(processed_data_file_test))
    df = pd.read_excel('/home/yt/code/MMFSyn/predictor/data/' + 'synergy.xlsx')
    train_druga, train_drugb, train_cell,  train_Y, train_tissue = list(df['drug_a']),list(df['drug_b']), list(df['cell_line']),list(df['score']),list(df['tissue'])
    #XT = [seq_cat(t) for t in train_prots]
    train_druga, train_drugb, train_cell,  train_Y, train_tissue = np.asarray(train_druga), np.asarray(train_drugb), np.asarray(train_cell), np.asarray(train_Y), np.asarray(train_tissue)


    # make data PyTorch Geometric ready
    print('preparing ', 'train.pt in pytorch format!') #dataset + 
    train_data = TestbedDataset(root='/home/yt/code/MMFSyn/predictor/data', dataset='synergy', xd1=train_druga, xd2=train_drugb, xt=train_cell, y=train_Y,tissue=train_tissue,smile_graph=smile_graph)
    print('preparing ', 'test.pt in pytorch format!')
    print(processed_data_file_train, ' have been created')
  else:
    print(processed_data_file_train, ' are already created')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Creation of dataset")
  parser.add_argument("--dataset",type=str,default='kiba',help="Dataset Name (davis,kiba,DTC,Metz,ToxCast,Stitch)")
  args = parser.parse_args()
  print(args)
  main(args)

