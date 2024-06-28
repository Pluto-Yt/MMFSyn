import torch
from torch_geometric.utils import to_dense_adj
import torch_geometric.utils as u
from scipy import sparse
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv,TransformerConv, global_max_pool as gmp
import math
import numpy as np
from torch_geometric.data.batch import Batch

from models.g_mlp_pytorch import gMLP

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AE2(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(AE2, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + 500 // 2) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + 500 // 2) // 2)

        self.att2 = EncoderLayer((self.vector_size + 500 // 2) // 2, 8)
        self.l2 = torch.nn.Linear((self.vector_size + 500 // 2) // 2, 500 // 2)

        self.l3 = torch.nn.Linear(500 // 2, (self.vector_size + 500 // 2) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + 500 // 2) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + 500 // 2) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(0.3)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]

        X1 = self.dr(self.ac(self.bn1((self.l1(X1)))))
        X1 = self.att2(X1)
        X1 = self.l2(X1)
        X_AE1 = self.dr(self.ac(self.bn3((self.l3(X1)))))
        X_AE1 = self.l4(X_AE1)

        X2 = self.dr(self.ac(self.bn1((self.l1(X2)))))
        X2 = self.att2(X2)
        X2 = self.l2(X2)
        X_AE2 = self.dr(self.ac(self.bn3((self.l3(X2)))))
        X_AE2 = self.l4(X_AE2)

        X = torch.cat((X1, X2), 1)
        X_AE = torch.cat((X_AE1, X_AE2), 1)

        return X, X_AE

class cov(torch.nn.Module):
    def __init__(self, vector_size, len_after_AE=500, cov2KerSize=50, cov1KerSize=25):
        super(cov, self).__init__()

        self.vector_size = vector_size

        self.cov2KerSize = cov2KerSize
        self.cov1KerSize = cov1KerSize
        self.len_after_AE = len_after_AE
        self.co2_1 = torch.nn.Conv2d(1, 1, kernel_size=(2, cov2KerSize))
        self.co1_1 = torch.nn.Conv1d(1, 1, kernel_size=cov1KerSize)
        self.pool1 = torch.nn.AdaptiveAvgPool1d(len_after_AE)

        self.ac = gelu

    def forward(self, X):
        X1 = X[:, 0:self.vector_size // 2]
        X2 = X[:, self.vector_size // 2:]

        X = torch.cat((X1, X2), 0)

        X = X.view(-1, 1, 2, self.vector_size // 2)

        X = self.ac(self.co2_1(X))

        X = X.view(-1, self.vector_size // 2 - self.cov2KerSize + 1, 1)
        X = X.permute(0, 2, 1)
        X = self.ac(self.co1_1(X))

        X = self.pool1(X)

        X = X.contiguous().view(-1, self.len_after_AE)

        return X

class ADDAE(torch.nn.Module):
    def __init__(self, vector_size):
        super(ADDAE, self).__init__()

        self.vector_size = vector_size // 2

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + 500) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + 500) // 2)

        self.att1 = EncoderLayer((self.vector_size + 500) // 2, 8)
        self.l2 = torch.nn.Linear((self.vector_size + 500) // 2, 500)
        self.att2=EncoderLayer(500//2,8)

        self.l3 = torch.nn.Linear(500, (self.vector_size + 500) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + 500) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + 500) // 2, self.vector_size *2)

        self.dr = torch.nn.Dropout(0.3)

        # self.vector_size = vector_size // 2

        # self.l1 = torch.nn.Linear(self.vector_size, self.vector_size * 4)
        # self.bn1 = torch.nn.BatchNorm1d(self.vector_size *4)

        # self.att1 = EncoderLayer(self.vector_size * 4, 8)
        # self.l2 = torch.nn.Linear(self.vector_size *4, self.vector_size *2)
        # self.att2=EncoderLayer(self.vector_size *2, 8)

        # self.l3 = torch.nn.Linear(self.vector_size*2 , self.vector_size *2)
        # self.bn3 = torch.nn.BatchNorm1d(self.vector_size *2)

        # self.l4 = torch.nn.Linear(self.vector_size *2, self.vector_size*2)

        # self.dr = torch.nn.Dropout(0.3)

        self.ac = gelu
    def forward(self, X):
        X1 = X[:, 0:self.vector_size]
        X2 = X[:, self.vector_size:]
        X = X1 + X2

        X = self.dr(self.ac(self.bn1(self.l1(X))))

        X = self.att1(X)
        X = self.l2(X)

        X_AE = self.dr(self.ac(self.bn3(self.l3(X))))

        X_AE = self.l4(X_AE)
        #X_AE = torch.cat((X_AE, X_AE), 1)

        return X, X_AE

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=1, bias=False) #1
        self.conv7 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv9 = nn.Conv1d(128, 128, kernel_size=1, bias=False) #1
        self.conv10 = nn.Conv1d(128, 160, kernel_size=1, bias=False)
        #self.conv10 = nn.Conv1d(200, 512, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)

        self.bn10 = nn.BatchNorm1d(160)
        #self.bn10 = nn.BatchNorm1d(512)

        self.linear1 = nn.Linear(160, 256, bias=False)
        self.bn11 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(256, 256)
        
        #self.bn7 = nn.BatchNorm1d(1024)
        #self.linear3 = nn.Linear(1024, 512)
        #self.attention3 = EncoderLayer(200, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + x1
        x = F.relu(self.bn4(self.conv4(x)))
        x2 = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = x + x2
        x = F.relu(self.bn7(self.conv7(x)))
        x3 = x
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = x + x3
        x = F.relu(self.bn10(self.conv10(x)))
        #x = F.relu(self.bn8(self.conv8(x)))
        #x = F.relu(self.bn10(self.conv10(x)))

        x = F.adaptive_max_pool1d(x, 1).squeeze()
        #x = self.attention3(x)
        x = F.relu(self.bn11(self.linear1(x)))
        #x = F.relu(self.bn7(self.linear2(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class CrossAttention(nn.Module):  
    def __init__(self, embed_dim, num_heads):  
        super(CrossAttention, self).__init__()  
        self.embed_dim = embed_dim  
        self.num_heads = num_heads  
        self.head_dim = embed_dim // num_heads  
          
        assert (self.head_dim * num_heads == self.embed_dim), "Embedding dimension must be divisible by num_heads"  
          
        self.query_linear = nn.Linear(self.embed_dim, self.embed_dim)  
        self.key_linear = nn.Linear(self.embed_dim, self.embed_dim)  
        self.value_linear = nn.Linear(self.embed_dim, self.embed_dim)  
        self.out_linear = nn.Linear(self.embed_dim, self.embed_dim)  
          
        self.softmax = nn.Softmax(dim=-1)  
          
    def split_heads(self, x, batch_size):  
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)  
        return x.permute(0, 2, 1, 3)  
      
    def forward(self, query, key, value, mask=None):  
        batch_size = query.shape[0]  
          
        query = self.query_linear(query)  
        key = self.key_linear(key)  
        value = self.value_linear(value)  
          
        query = self.split_heads(query, batch_size)  
        key = self.split_heads(key, batch_size)  
        value = self.split_heads(value, batch_size)  
          
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key]) / self.head_dim ** 0.5  
          
        if mask is not None:  
            energy = energy.masked_fill(mask == 0, float('-1e20'))  
          
        attention = self.softmax(energy)  
          
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(batch_size, -1, self.embed_dim)  
        out = self.out_linear(out)  
          
        return out  



class MultiHeadAttention1(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention1, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
        self.trans = torch.nn.Linear(1920*2, 1920)
    def forward(self, X, Y, Z):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(Y).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(self.trans(Z)).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class EncoderLayer1(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer1, self).__init__()
        self.attn = MultiHeadAttention1(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X,Y,Z):
        output = self.attn(X,Y,Z)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X

class MMFNet(torch.nn.Module):
  def __init__(self,k1,k2,k3,embed_dim,num_layer,device,num_feature_xd=78,n_output=1,num_feature_xt=25,output_dim=128,dropout=0.2):
    super(MMFNet,self).__init__()
    self.device = device
    # Smile graph branch
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.embed_dim = embed_dim
    self.num_layer = num_layer
    self.Conv1 = GCNConv(num_feature_xd,num_feature_xd)
    self.Conv2 = GCNConv(num_feature_xd,num_feature_xd*2)
    self.Conv3 = GCNConv(num_feature_xd*2,num_feature_xd*4)
    self.Conv4 = GCNConv(num_feature_xd*4,num_feature_xd*4)
    self.Conv5 = GCNConv(num_feature_xd*4,num_feature_xd*2)
    self.Conv6 = GCNConv(num_feature_xd*2,num_feature_xd)

    self.relu = nn.ReLU()
    self.fc_g1 = nn.Linear(546,1024)
    self.fc_g2 = nn.Linear(1024,output_dim)
    self.dropout = nn.Dropout(dropout)

    #protien sequence branch (LSTM)
    self.embedding_xt = nn.Embedding(num_feature_xt+1,embed_dim)
    self.LSTM_xt_1 = nn.LSTM(self.embed_dim,self.embed_dim,self.num_layer,batch_first = True,bidirectional=True)
    # self.fc_xt = nn.Linear(1000*256,output_dim)
    self.gmlp = gMLP(num_tokens = None,dim = 78*2,depth = 10,seq_len = 160,causal = True,circulant_matrix = True,heads = 3,attn_dim=3) #
    self.ln1 = nn.Linear(160*78*2, 256, bias=False)
    self.bnn1 = nn.BatchNorm1d(256)
    self.dpp1 = nn.Dropout()
    self.ln2 = nn.Linear(256, 256)

    #combined layers
    self.fc1 = nn.Linear(2*output_dim+512,1024)
    self.fc2 = nn.Linear(1024,512)
    self.out = nn.Linear(512,n_output)
    self.hidden_size = 8192
    self.network = nn.Sequential(
            nn.Linear(1920, self.hidden_size), #5048+128-3384+14091//4-512+128 #5058
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, n_output)
    )

    self.cellwork = nn.Sequential(
            nn.Linear(14091, 14091 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(14091 // 2),
            nn.Linear(14091 // 2, 1920) #14091 // 4
    )
    self.pointnet = PointNet()

    self.att1=EncoderLayer(384+256*2, 4)
    self.ADDAE = ADDAE(768-256)
    self.ADDAE2 = ADDAE(256*2)
    self.sigmoid = nn.Sigmoid()
    self.CrossAttention = CrossAttention(1920, 1)
    self.MultiHeadAttention1 = MultiHeadAttention1(1920, 1)

    self.EncoderLayer1 = EncoderLayer1(1920,4)
  def forward(self,data): #,hidden,cell
    #x1 , edge_index1, batch1 = data.x,data.edge_index,data.batch
    data11 = data.graph1
    data22 = data.graph2
    data1 = Batch.from_data_list(data11)#调用该函数data_list里的data1、data2、data3 三张图形成一张大图，也就是batch
    data2 = Batch.from_data_list(data22)#调用该函数data_list里的data1、data2、data3 三张图形成一张大图，也就是batch
    
    x1 , edge_index1, batch1 = data1.x,data1.edge_index,data1.batch
    x2 , edge_index2, batch2 = data2.x,data2.edge_index,data2.batch

    
    #xx = [data1,data2]
    #datadata = Batch.from_data_list(xx)
    x = data.x
    batch = data.batch
    #edge_index = torch.cat((edge_index1,edge_index2),1)
    edge_index = data.edge_index
    adj1 = to_dense_adj(edge_index1)
    adj2 = to_dense_adj(edge_index2)

    target_ge = data.target_ge
    target_mut = data.target_mut
    target = torch.cat((target_ge,target_mut),1)
    drug1_3D = data.drug1_3D
    drug1_morgan = data.drug1_morgan
    drug1_2D = data.drug1_2D
    drug2_3D = data.drug2_3D
    drug2_morgan = data.drug2_morgan
    drug2_2D = data.drug2_2D

    if self.k1 == 1:
      h11 = self.Conv1(x1,edge_index1)

      h11 = self.relu(h11)

      h12 = self.Conv2(h11,edge_index1)
  
      h12 = self.relu(h12)

      h13 = self.Conv3(h12,edge_index1)

      h13 = self.relu(h13)

    if self.k2 == 2:
      edge_index_square1,_ = torch_sparse.spspmm(edge_index1,None,edge_index1,None,adj1.shape[1],adj1.shape[1],adj1.shape[1],coalesced=True)
      h14 = self.Conv1(x1,edge_index_square1)
      h14 = self.relu(h14)
      h15 = self.Conv2(h14,edge_index_square1)
      h15 = self.relu(h15)

    if self.k3 == 3:
      edge_index_cube1,_ = torch_sparse.spspmm(edge_index_square1,None,edge_index1,None,adj1.shape[1],adj1.shape[1],adj1.shape[1],coalesced=True)
      h16 = self.Conv1(x1,edge_index_cube1)
      h16 = self.relu(h16)
    
    concat1 = torch.cat([h13,h15,h16],dim=1)

    x1 = gmp(concat1,batch1) #global_max_pooling
    #flatten
    x1 = self.relu(self.fc_g1(x1))
    x1 = self.dropout(x1)
    x1 = self.fc_g2(x1)
    x1 = self.dropout(x1)


    #处理x2
    if self.k1 == 1:
      h21 = self.Conv1(x2,edge_index2)

      h21 = self.relu(h21)

      h22 = self.Conv2(h21,edge_index2)
  
      h22 = self.relu(h22)

      h23 = self.Conv3(h22,edge_index2)

      h23 = self.relu(h23)

    if self.k2 == 2:
      edge_index_square2,_ = torch_sparse.spspmm(edge_index2,None,edge_index2,None,adj2.shape[1],adj2.shape[1],adj2.shape[1],coalesced=True)
      h24 = self.Conv1(x2,edge_index_square2)
      h24 = self.relu(h24)
      h25 = self.Conv2(h24,edge_index_square2)
      h25 = self.relu(h25)

    if self.k3 == 3:
      edge_index_cube2,_ = torch_sparse.spspmm(edge_index_square2,None,edge_index2,None,adj2.shape[1],adj2.shape[1],adj2.shape[1],coalesced=True)
      h26 = self.Conv1(x2,edge_index_cube2)
      h26 = self.relu(h26)

    concat2 = torch.cat([h23,h25,h26],dim=1)

    x2 = gmp(concat2,batch2) #global_max_pooling
    #flatten
    x2 = self.relu(self.fc_g1(x2))
    x2 = self.dropout(x2)
    x2 = self.fc_g2(x2)
    x2 = self.dropout(x2)

    #处理x1,x2
    if self.k1 == 1:
      hd1 = self.Conv1(x,edge_index)

      hd1 = self.relu(hd1)

      hd2 = self.Conv2(hd1,edge_index)
  
      hd2 = self.relu(hd2)

      hd3 = self.Conv3(hd2,edge_index)

      hd3 = self.relu(hd3)


    if self.k2 == 2:

      edge_index_square = data.edge_index_square #torch.cat((edge_index_square1, edge_index_square2),1)
      hd11 = self.Conv1(x,edge_index_square)
      hd11 = self.relu(hd11)
      hd22 = self.Conv2(hd11,edge_index_square)
      hd22 = self.relu(hd22)

    if self.k3 == 3:
      edge_index_cube = data.edge_index_cube
      hd66 = self.Conv1(x,edge_index_cube)
      hd66 = self.relu(hd66)

    concat = torch.cat([hd3,hd22,hd66],dim=1)

    x = gmp(concat,batch) #global_max_pooling
    #flatten
    x = self.relu(self.fc_g1(x))
    x = self.dropout(x)
    x = self.fc_g2(x)
    x = self.dropout(x)

    # #处理二维特征
    LSTM_xt1,_ = self.LSTM_xt_1(drug1_2D) #,(hidden,cell)
    drug1_2D_x = self.gmlp(LSTM_xt1)
    drug1_2D_x = drug1_2D_x.view(-1,160*78*2)
    drug1_2D_x = self.ln2(self.dpp1(self.bnn1(self.ln1(drug1_2D_x))))

    LSTM_xt2,_ = self.LSTM_xt_1(drug2_2D)
    drug2_2D_x = self.gmlp(LSTM_xt2)
    drug2_2D_x = drug2_2D_x.view(-1,160*78*2)
    drug2_2D_x = self.ln2(self.dpp1(self.bnn1(self.ln1(drug2_2D_x))))

    # pointnet layer
    drug1_3D = drug1_3D.transpose(1,2)
    drug1_3D = self.pointnet(drug1_3D)

    drug2_3D = drug2_3D.transpose(1,2)
    drug2_3D = self.pointnet(drug2_3D)

    #concat
    target = self.cellwork(target)

    xxxx = torch.cat((drug1_2D_x,drug2_2D_x),1)
    _,xxxx = self.ADDAE(xxxx)
    xc = torch.cat((drug1_morgan,drug1_3D,x1,xxxx,x,drug2_morgan,drug2_3D,x2,target),1)
    xc1 = torch.cat((drug1_morgan,drug1_3D,x1,xxxx,x,drug2_morgan,drug2_3D,x2),1)

    xcc = self.EncoderLayer1(xc1, target, xc)
    out = self.network(xcc)
    return out