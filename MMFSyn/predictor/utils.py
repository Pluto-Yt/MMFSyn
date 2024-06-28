import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch

from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

from rdkit.ML.Descriptors import MoleculeDescriptors

from data_creation import smile_to_graph
import pandas as pd

from torch_geometric.utils import to_dense_adj
import torch_sparse

def smilesto3D(self,path,smiles):
    mol = Chem.MolFromMolFile(path)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=10, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    Chem.MolToXYZFile(mol, '/home/yt/code/DeepGLSTM/data/drugsmiles3D/{}.txt'.format(smiles))

    f = open('/home/yt/code/DeepGLSTM/data/drugsmiles3D/{}.txt'.format(smiles), 'rb')
    lines = f.readlines()
    list = []
    for line in lines:
        #line = line.decode('gb2312').encode('utf8') 
        list.append(line)
    del list[0:2]
    #print(list)
    list2 = []
    for i in list:
        i = str(i).replace("\\n'", '')
        i = str(i).split()
        list2.append(i[1:4])
    A = np.array(list2)
    A = A.astype(np.float).tolist()
    C= [0,0,0]
    CC = []
    ccc= 160-len(A)
    for i in range(ccc):
        A.append(C)
    # for i in range(6):
    #     A += augment_data(A,30, 0.01, 0.05).tolist()
    B = torch.Tensor(A[0:160])
    B = torch.unsqueeze(B, 0)
    B = torch.Tensor(B).unsqueeze(0)
    # print(B.shape)
    # torch.save(B, '/home/yt/code/Molormer/dataset/3D-drug/{}_3D.pt'.format(id1))

    return B

def smilestomorgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
    A = list(fingerprints.ToBitString()) # result 2
    A = map(float, A)
    AA = list(A)
    B = torch.Tensor(AA).unsqueeze(0)
    return B

def buquansmilesxulie(smiles):
    #smiles1 = Chem.MolFromSmiles(smiles)
    c_size, features, edge_index = smile_to_graph(smiles)
    drug_2D = []
    lenth = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    bulenth= 100-len(features)
    for z in range(bulenth):
        features.append(lenth)
    drug_2D.append(features)
    drug_2D = torch.Tensor(features).unsqueeze(0)
    return drug_2D

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='', 
                 xd1=None, xd2=None, xt=None, y=None,tissue=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd1,xd2, xt, y,tissue,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd1, xd2, xt, y,tissue,smile_graph):
        assert (len(xd1) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        
        data_list = []
        data_len = len(xd1)
        dd = pd.read_csv('/home/yt/code/MMFSyn/drug/data/smiles.csv')
        data_smile = dict(zip(dd['name'],dd['smile']))

        cd = pd.read_excel('/home/yt/code/MMFSyn/cell/data/cell2id.xlsx')
        cell2id = dict(zip(cd['cell'],cd['id']))
        cell_ge = np.load('/home/yt/code/MMFSyn/cell/data/target_ge.npy')
        cell_mut = np.load('/home/yt/code/MMFSyn/cell/data/target_mut.npy')

        ddd = pd.read_excel('/home/yt/code/MMFSyn/drug/data/drug2id.xlsx')
        drug2id = dict(zip(ddd['drug'],ddd['id']))

        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles1 = data_smile[xd1[i]]
            smiles2 = data_smile[xd2[i]]
            target_ge = cell_ge[cell2id[xt[i]]]
            target_mut = cell_mut[cell2id[xt[i]]]
            drug_A = xd1[i]
            drug_B = xd2[i]
            cell_line = xt[i]
            tissue1 = tissue[i]
            labels = y[i]
            drug1_3D = torch.load('/home/yt/code/MMFSyn/drug/data/3Ddrug_tensor6-160/drug{}_3D.pt'.format(drug2id[xd1[i]])) #torch.load('/home/yt/code/DeepGLSTM/data/kiba_3D/{}.pt'.format(smiles1))
            drug2_3D = torch.load('/home/yt/code/MMFSyn/drug/data/3Ddrug_tensor6-160/drug{}_3D.pt'.format(drug2id[xd2[i]])) #torch.load('/home/yt/code/DeepGLSTM/data/kiba_3D/{}.pt'.format(smiles1))

            drug1_2D = torch.load('/home/yt/code/MMFSyn/drug/data/2Ddrug_tensor1-160/drug{}_2D.pt'.format(drug2id[xd1[i]])) #smilestomorgan(str(smiles))
            drug2_2D = torch.load('/home/yt/code/MMFSyn/drug/data/2Ddrug_tensor1-160/drug{}_2D.pt'.format(drug2id[xd2[i]])) #smilestomorgan(str(smiles))

            drug1_morgan = smilestomorgan(str(smiles1))
            drug2_morgan = smilestomorgan(str(smiles2))

            # convert SMILES to molecular representation using rdkit
            c_size1, features1, edge_index1 = smile_graph[smiles1]
            c_size2, features2, edge_index2 = smile_graph[smiles2]

            x1=features1
            edge_index1=(edge_index1).transpose(1, 0)
            x2=features2
            edge_index2=(edge_index2).transpose(1, 0)
            x = torch.cat((x1,x2),0)
            edge_index = torch.cat((edge_index1,edge_index2),1)
            adj1 = to_dense_adj(edge_index1)
            adj2 = to_dense_adj(edge_index2)
            edge_index_square1,_ = torch_sparse.spspmm(edge_index1,None,edge_index1,None,adj1.shape[1],adj1.shape[1],adj1.shape[1],coalesced=True)
            edge_index_square2,_ = torch_sparse.spspmm(edge_index2,None,edge_index2,None,adj2.shape[1],adj2.shape[1],adj2.shape[1],coalesced=True)
            edge_index_square = torch.cat((edge_index_square1, edge_index_square2),1)
            edge_index_cube1,_ = torch_sparse.spspmm(edge_index_square1,None,edge_index1,None,adj1.shape[1],adj1.shape[1],adj1.shape[1],coalesced=True)
            edge_index_cube2,_ = torch_sparse.spspmm(edge_index_square2,None,edge_index2,None,adj2.shape[1],adj2.shape[1],adj2.shape[1],coalesced=True)
            edge_index_cube = torch.cat((edge_index_cube1, edge_index_cube2),1)
            GCNData = DATA.Data(x=x,edge_index=torch.LongTensor(edge_index),y=torch.FloatTensor([labels]))
            GCNData.edge_index_square = torch.LongTensor(edge_index_square)
            GCNData.edge_index_cube = torch.LongTensor(edge_index_cube)

            GCNData.graph1 = DATA.Data(x=x1,edge_index=torch.LongTensor(edge_index1))
            GCNData.graph2 = DATA.Data(x=x2,edge_index=torch.LongTensor(edge_index2))
            # GCNData.x1 = torch.LongTensor(features1)
            # GCNData.x2 = torch.Tensor(features2)
            # # GCNData.edge1 = torch.LongTensor(edge_index1).transpose(1, 0)
            # GCNData.edge2 = torch.LongTensor(edge_index2)

            # GCNData.x2 = GCNData2.x
            # GCNData.edge2 = GCNData2.edge_index
            # GCNData.batch2 = GCNData2.batch

            GCNData.target_ge = torch.Tensor([target_ge])
            GCNData.target_mut = torch.Tensor([target_mut])

            GCNData.drug1_3D = drug1_3D
            GCNData.drug2_3D = drug2_3D

            GCNData.drug1_morgan = drug1_morgan
            GCNData.drug2_morgan = drug2_morgan

            GCNData.drug1_2D = drug1_2D
            GCNData.drug2_2D = drug2_2D

            GCNData.drug_A = drug_A
            GCNData.drug_B = drug_B
            GCNData.cee_line = cell_line
            GCNData.tissue = tissue1

            #GCNData.y = torch.FloatTensor([labels])

            # GCNData.__setitem__('c_size1', torch.LongTensor([c_size1]))
            # GCNData.__setitem__('c_size2', torch.LongTensor([c_size2]))

            # append graph, label and target sequence to data list
            data_list.append(GCNData)



        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    ys_orig = np.concatenate(ys_orig)
    ys_line = np.concatenate(ys_line)
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : dataset_collate.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-07 17:09:13
"""
 
r""""Contains definitions of the methods used by the _DataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).
These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""
 
import torch
import re
from torch._six import  string_classes
import collections.abc as container_abcs
int_classes = int

_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""
 
np_str_obj_array_pattern = re.compile(r'[SaUO]')
 
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
 
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
 
 
def collate_fn(batch):
    '''
     collate_fn (callable, optional): merges a list of samples to form a mini-batch.
     该函数参考touch的default_collate函数，也是DataLoader的默认的校对方法，当batch中含有None等数据时，
     默认的default_collate校队方法会出现错误
     一种的解决方法是：
     判断batch中image是否为None,如果为None，则在原来的batch中清除掉，这样就可以在迭代中避免出错了
    :param batch:
    :return:
    '''
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # 这里添加：判断image是否为None,如果为None，则在原来的batch中清除掉，这样就可以在迭代中避免出错了
    if isinstance(batch, list):
        batch = [(image, image_id) for (image, image_id) in batch if image is not None]
    if batch==[]:
        return (None,None)
 
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))
 
            return collate_fn([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)#ok
        return [collate_fn(samples) for samples in transposed]
 
    raise TypeError((error_msg_fmt.format(type(batch[0]))))