from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
import scipy
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from fpgnn.data import GetPubChemFPs,prepare_feature, create_graph, get_atom_features_dim
import csv
import os
import sys
import time
import math
import copy
import pandas as pd

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

import pickle

#from dgl import functional as fn

atts_out = []


class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):   
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:   # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads    # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)   
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变
        
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size,bias=False)    # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size,bias = False)
        self.value = nn.Linear(hidden_size, self.all_head_size,bias = False)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   # 
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]

    def forward(self, hidden_states, attention_mask=None):

        # 线性变换
        mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)    # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)   # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer+hidden_states    # [bs, seqlen, 128] 得到输出

class FPAN(nn.Module):

    def __init__(self,args):
        super(FPAN,self).__init__()
        self.args = args
        
        self.input_dim=512

        self.maccslen=167
        self.pubchemlen=881
        self.ecfplen=1024
        self.erglen=441
        self.maxsize=896
        self.num_heads=8

        self.attention=SelfAttention(self.maxsize,8,args.sadp)
        #self.attention2=SelfAttention(self.maxsize,8,args.sadp)

        self.fp_2_dim=args.hidden_size
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size

        self.smile2fp= {}

        self.fc1=nn.Linear(3*self.maxsize, self.hidden_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smile):
        fpbatch=torch.empty(size=(len(smile),3,self.maxsize))
        fpbatch = fpbatch.cuda()
        for i, one in enumerate(smile):

            if one in self.smile2fp:
                fps=self.smile2fp[one]
            else:
                fps=[]
                mol = Chem.MolFromSmiles(one)
                fp_maccs = list(AllChem.GetMACCSKeysFingerprint(mol))+[0]*(self.maxsize-self.maccslen)
                fp_erg = list(AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1))+[0]*(self.maxsize-self.erglen)
                fp_pubchem = list(GetPubChemFPs(mol))+[0]*(self.maxsize-self.pubchemlen)
                #fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.extend(fp_maccs+fp_erg+fp_pubchem)

                self.smile2fp[one]=fps
            fp_tensor=torch.Tensor(fps).view(3,-1)

        
            if self.cuda:
                fp_tensor=fp_tensor.cuda()
 
            fpbatch[i]=fp_tensor

        #print(query.device)
        fpn_out= self.attention(fpbatch)
        #fpn_out= self.attention2(fpn_out)
        
        fpn_out = fpn_out.view(-1,3*self.maxsize)

        fpn_out = self.fc1(fpn_out)
        return fpn_out



class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        mode = "mul",
        att = "edge",
        feat_drop=0.,
        attn_drop=0.,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_feats = 8
        self._allow_zero_in_degree = allow_zero_in_degree
        self.elength = out_feats
        self.mode = mode
        if mode == "cross":
            self.embeding = 3
        else:
            self.embeding = 32
        
        self.att = att
        #self.belta = nn.Parameter(torch.FloatTensor(1),requires_grad = True)


        #self.fc = nn.Linear( self._in_src_feats, out_feats * num_heads, bias=False)

        self.q = nn.Linear( self.in_feats, self.embeding * num_heads, bias = False)
        self.k = nn.Linear( self.in_feats, self.embeding * num_heads, bias = False)
        self.v = nn.Linear( self.in_feats, out_feats * num_heads,bias=False)

        self.attfc = nn.Linear(self.in_feats,self.num_heads,bias = True)


        
        self.edge_fc = nn.Linear(self.edge_feats,self.num_heads*self.out_feats,bias=True)

        self.edge_k = nn.Linear(self.edge_feats,self.num_heads*self.embeding,bias=True)
        #self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        #self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self.in_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self.in_feats, num_heads * out_feats, bias=bias)
                self.has_linear_res = True
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation
        self.act = nn.functional.elu
        #self.act =  torch.sigmoid
        #self.activation = torch.nn.functional.elu
        

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.v.weight, gain=gain)
        nn.init.xavier_normal_(self.q.weight, gain=gain)
        nn.init.xavier_normal_(self.k.weight, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value
    
    def edge_function(self,edges):
        crossproduct = torch.cross(edges.src['query'],edges.dst['key'])
        epoint = (crossproduct*edges.data['edge_key']).sum(dim=-1).unsqueeze(-1)
        return {'e':epoint}

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            n = feat.shape[0]

            h_src = h_dst = self.feat_drop(feat)
            

            feat_src  = self.v(h_src).view(
                n, self.num_heads, self.out_feats
                )
            #el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            #er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            edge_feat = graph.edata['bond']
            edge_key = self.edge_k(edge_feat).view(-1,self.num_heads,self.embeding)
            graph.edata.update({"edge_key":edge_key})
            #if self.use_act:
            #h_src = nn.functional.elu(feat)
            src_embeding = self.q(h_src)
            dst_embeding = self.k(h_src)

            query = src_embeding.view(n,self.num_heads,self.embeding)
            key = dst_embeding.view(n,self.num_heads,self.embeding)
            #e=(query*key).sum(dim=-1).unsqueeze(-1)
            graph.ndata.update({"ft":feat_src,"query":query,"key":key})

            #graph.ndata.update({"ft": feat_src, "ek": el})
            #graph.ndata.update({"er": er})]
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            #*****graph.apply_edges(fn.u_mul_v("query", "key", "e"))
            #graph.apply_edges(fn.u_dot_v("query", "key", "e"))
            #***graph.edata['e'] = (graph.edata['e']*graph.edata['edge_key']).sum(dim=-1).unsqueeze(-1)
            #e = self.leaky_relu(graph.edata.pop("e").sum(dim=-1).unsqueeze(-1))

            #e = self.leaky_relu(graph.edata.pop("e").unsqueeze(-1))
            
            """
            mul   (a*b)*c
            add   (a+b)*c
            cross   (axb)*c
            """
            if self.mode =="cross":
                graph.apply_edges(self.edge_function)
            elif self.mode == "mul":
                graph.apply_edges(fn.u_mul_v("query", "key", "e"))
                graph.edata['e'] = (graph.edata['e']*graph.edata['edge_key']).sum(dim=-1).unsqueeze(-1)
            else:
                graph.apply_edges(fn.u_add_v("query","key","e"))
                graph.edata['e'] = (graph.edata['e']*graph.edata['edge_key']).sum(dim=-1).unsqueeze(-1)

            # compute softmax
            #e = graph.edata.pop("e")
            #e = self.leaky_relu(e)
            e = graph.edata["e"]
            #sse= torch.clamp(e,-1000.,1.414)
            #
            #e= torch.clamp(e,0,2)
            #
            #graph.edata["a"] = e

            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
                

            #print("a size",graph.edata['a'].size())
            #print("a tensor",graph.edata['a'])
            # message passing
            
            bond_feat = graph.edata.pop("bond")
            edge_feat = self.edge_fc(bond_feat).view(-1,self.num_heads,self.out_feats)
            graph.edata.update({"edge_feat":edge_feat})

            if self.att == "total":


                """
                src_embeding = self.act(src_embeding)
                dst_embeding = self.act(dst_embeding)
                bond_src = self.bond_left_fc(src_embeding).view(-1,self.num_heads,self.elength)
                bond_dst = self.bond_right_fc(dst_embeding).view(-1,self.num_heads,self.elength)
                graph.ndata.update({"bond_src":bond_src,"bond_dst":bond_dst})
                

                #graph.apply_edges(fn.v_mul_u("bond_src", "bond_dst", "x"))
                #graph.apply_edges(fn.v_add_u("bond_src", "bond_dst", "x"))
                graph.apply_edges(fn.u_dot_v("bond_src", "bond_src", "x"))
                #graph.apply_edges(fn.u_sub_v("bond_src", "bond_dst", "x"))
                #graph.ndata["zeros"] = torch.zeros_like(graph.ndata["bond_src"])

                #graph.apply_edges(fn.u_add_v("zeros","bond_src","x"))
                #graph.apply_edges(fn.e_add_v("x","bond_src","x"))
                #graph.apply_edges(fn.u_add_v( "edge_feat","bond_key", "edge_feat"))
                #graph.apply_edges(fn.copy_src("bond_dst","x"))

                """

                edge_src = self.bond_src(feat).view(n,self.num_heads,self.elength)
                edge_dst = self.bond_dst(feat).view(n,self.num_heads,self.elength)
                graph.ndata.update({"bond_src":edge_src,"bond_dst":edge_dst})

                graph.apply_edges(fn.u_dot_v("bond_src", "bond_dst", "x"))
                #graph.apply_edges(fn.u_dot_v("bond_src", "bond_dst", "x"))

                graph.edata["edge_feat"] = graph.edata["edge_feat"]*graph.edata["x"]






            #graph.apply_edges(fn.u_add_e("ft", "edge_feat", "message"))
            #
            graph.apply_edges(fn.u_mul_e("ft", "edge_feat", "message"))
            graph.edata['message'] = graph.edata['message']*graph.edata['a']
            graph.update_all(fn.copy_e("message","m"), fn.sum("m", "ft"))
            

            #graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.ndata["ft"]
            #feat = nn.functional.relu(feat)
            #recpoint = self.attfc(feat).unsqueeze(-1)
            #recpoint = self.attfc(feat).view(n,self.num_heads,-1)
            #rst = rst*recpoint
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    n, -1, self.out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    -1,
                    self.num_heads,
                    self.out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst

class MPNN(nn.Module):
    def __init__(self,in_features,out_features):
        super(MPNN,self).__init__()
        self.fc1=nn.Linear(in_features,out_features,bias = True)
        self.fc2=nn.Linear(out_features,out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.elu = nn.functional.elu

        self.attfc = nn.Linear(in_features,out_features,bias = True)
        #self.attfc = nn.Linear(in_features,out_features,bias = True)
        self.belta = nn.Parameter(torch.FloatTensor(1),requires_grad = True)
        #self.belta = 1
        #self.fc3 = nn.Linear(out_features,out_features)

        self.embeding = 32
        self.fc_src = nn.Linear(in_features,self.embeding,bias =True)
        self.fc_dst = nn.Linear(in_features,self.embeding,bias =True)
        
    
    def forward(self,graph,feat):
        with graph.local_scope():
            feat_src=self.fc1(feat)
            #feat_src=self.leaky_relu(feat_src)
            #feat_src=nn.functional.relu(feat_src)
            
            #feat_src=self.elu(feat_src)
            graph.ndata.update({"feat_src":feat_src})
            
            
            src_embeding = self.fc_src(feat)
            dst_embeding = self.fc_dst(feat)
            graph.ndata.update({"src_emb":src_embeding,"dst_emb":dst_embeding})
            graph.apply_edges(fn.u_mul_v("src_emb", "dst_emb", "x"))
            graph.edata["x"] = graph.edata["x"].sum(dim=-1)
            #graph.edata["x"] = torch.clip(graph.edata["x"],-1,1)
            #graph.edata["a"] = torch.exp(graph.edata["x"])
            graph.edata["weight"] = (self.belta/graph.edata["dist"])*graph.edata["x"]
            #graph.edata["weight"] = edge_softmax(graph, graph.edata["weight"])

            
            
            #graph.edata["weight"] = (1/graph.edata["dist"])
            #graph.edata["weight"] = (self.belta/graph.edata["dist"])
            #print(graph.edata["weight"].size())


            #weight = self.belta/graph.edata.pop("dist")

            
            
            #graph.edata.update({"weight":weight})
            feat = nn.functional.relu(feat)
            graph.update_all(fn.u_mul_e("feat_src", "weight", "m"), fn.sum("m", "ft"))
            #graph.update_all(fn.u_mul_e("feat_src", "weight", "m"), fn.mean("m", "ft"))

            feat = nn.functional.relu(feat)
            e = self.attfc(feat)
            
            #e = nn.functional.leaky_relu((e),negative_slope = 0.1)
            graph.ndata['e'] = e
            #e= nn.functional.elu(e)

            #rst = graph.ndata["ft"]+(1+self.e)*feat
            #rst = graph.ndata["ft"]*(1-graph.ndata['e'])+graph.ndata['e']*feat
            rst = graph.ndata['e']*graph.ndata["ft"]#+feat
            #rst = graph.ndata["ft"]*graph.ndata['e']
            #rst = self.fc2(rst)
            #rst = self.leaky_relu(rst)
            #rst = nn.functional.relu(rst)
            rst = self.elu(rst)
            return rst
"""
class GNN(nn.Module):
    def __init__(self,args):
        super(GNN,self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.nheads = 8
        self.smile2graph={}
        self.smile2dist={}
        self.smile2edge = {}
        self.atom_feat_num = 32
        self.fc = nn.Linear(self.atom_feat_num*self.nheads*6,args.hidden_size,bias = True)
        self.sum=0

        self.gat = GATConv(89,self.atom_feat_num,self.nheads,
                    feat_drop = 0.0,
                    attn_drop = 0.,
                    mode = args.mode,
                    att = args.att,
                    #activation= nn.functional.elu,
                   )
        self.gat2 = GATConv(self.atom_feat_num*self.nheads,
                    self.atom_feat_num*self.nheads,1,
                    mode = args.mode,
                    att = args.att,
                    feat_drop=0.,attn_drop=0.,
                    residual = True,
                    activation= nn.functional.elu,
                    )
        self.mpnn = MPNN(self.atom_feat_num*self.nheads,
                   self.atom_feat_num*self.nheads
                )

        self.prepare_data()
        
    
    def prepare_data(self):
        #prepare_feature(self.args)
        data_path = self.args.data_path
        dataset = data_path.split("/")[-1][:-4]

        
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(os.path.join("data",'{}.graph'.format(dataset))):
            self.generate_graph(data_path)
            

        with open(os.path.join("data",'{}.graph'.format(dataset)),"rb") as f:
            self.smile2graph = pickle.load(f)
        
        with open(os.path.join("data",'{}.distgraph'.format(dataset)),"rb") as f:
            self.smile2dist = pickle.load(f)               


    def get_bond_feat(self,mol,graph,distmatrix):
        e = graph.number_of_edges()

        src,dst = graph.edges()
        edge_feats = []
        bond_type = {"SINGLE":1,"DOUBLE":2,"TRIPLE":3,"AROMATIC":4}
        for index in range(e):
            a = int(src[index])
            b = int(dst[index])
            bond_feat = [0]*7
            if a == b :
                bond_feat[0]=1
            else:
                bond = mol.GetBondBetweenAtoms(a,b)
                btype = str(bond.GetBondType())
                idx =  1 if btype not in bond_type else bond_type[btype]
                bond_feat[idx] = 1
                if bond.IsInRing():
                    bond_feat[5]=1
                dist = distmatrix[a][b]
                bond_feat[6] = dist
                #print(dist)
            edge_feats.append(bond_feat)

        return edge_feats

    def generate_graph(self,data_path):
        df = pd.read_csv(data_path)
        smiles = df.iloc[1:,0]
        dataset  = data_path.split("/")[-1][:-4]
        path = os.path.join("data","{}.feature".format(dataset))
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj[np.diag_indices_from(adj)] = 1
            cooadj = scipy.sparse.coo_matrix(adj)
            #self.smile2graph[smile] = graph

            if AllChem.EmbedMolecule(mol) == -1:
                Chem.rdmolops.RemoveStereochemistry(mol)
                if AllChem.EmbedMolecule(mol,useRandomCoords=True,maxAttempts=10) == -1 :
                    dist = Chem.GetDistanceMatrix(mol)*1.25
                    print("emebed failed,*****************************")
                else:
                    dist=AllChem.Get3DDistanceMatrix(mol)
            else:
                dist=AllChem.Get3DDistanceMatrix(mol)
            
            graph = dgl.from_scipy(cooadj)

            bond_feat = self.get_bond_feat(mol,graph,dist)
            edge_feat_tensor = torch.Tensor(bond_feat)
            graph.edata['bond'] = edge_feat_tensor
            self.smile2graph[smile] = graph

            hop2adj=np.linalg.matrix_power(adj,2)
            hop2adj[hop2adj>0]=1

            hopk=1-hop2adj
            dist[dist>5.]=0
            dist = hopk*dist*dist
            distweight = scipy.sparse.coo_matrix(dist)
            distgraph = dgl.from_scipy(distweight,eweight_name="dist")
            self.smile2dist[smile]=distgraph

        with open(os.path.join("data","{}.graph".format(dataset)),"wb") as f:
                pickle.dump(self.smile2graph,f)
            
        with open(os.path.join("data","{}.distgraph".format(dataset)),"wb") as f:
                pickle.dump(self.smile2dist,f)



    def forward(self,smiles):
        #smiles = ["Cc1cc(cc(c1)O)C"]
        mols = create_graph(smiles, self.args)
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()
        graphs = []
        distgraphs=[]
        for smile in smiles:
            graph = self.smile2graph[smile]
            distgraph = self.smile2dist[smile]            
            graphs.append(graph)
            distgraphs.append(distgraph)
       
        biggraph = dgl.batch(graphs).to("cuda")
        distgraph = dgl.batch(distgraphs).to("cuda")
        distgraph = dgl.to_float(distgraph)

        batch_feature = self.gat(biggraph,atom_feature[1:]).view(-1,self.atom_feat_num*self.nheads)
        biggraph.ndata['gat_feat']=self.gat2(biggraph,batch_feature).view(-1,self.atom_feat_num*self.nheads)
        #biggraph.ndata['gat_feat'] = batch_feature

        #gat_outs = torch.stack(gat_outs, dim=0)
        distgraph.ndata['mpnn_feat'] = self.mpnn(distgraph,biggraph.ndata['gat_feat'])
        gnn_out =torch.cat([dgl.sum_nodes(biggraph, 'gat_feat'),
                           dgl.max_nodes(biggraph, 'gat_feat'),
                           dgl.sum_nodes(distgraph, 'mpnn_feat'),
                           dgl.max_nodes(distgraph, 'mpnn_feat'),
                           dgl.mean_nodes(distgraph, 'mpnn_feat'),
                           dgl.mean_nodes(biggraph, 'gat_feat')
                           ],dim=-1)       


        #gnn_out =self.att(gnn_out)
        #gnn_out = gnn_out.view(-1,self.atom_feat_num*self.nheads*6)
        
        return self.fc(gnn_out)
"""



class GNN(nn.Module):
    def __init__(self,args):
        super(GNN,self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.nheads = 8
        self.smile2graph={}
        self.smile2dist={}
        self.smile2dist2 = {}
        self.smile2dist3 = {}
        self.smile2dist4 = {}
        self.smile2dist5 = {}
        self.smile2dist6 = {}

        self.atom_feat_num = args.atom
        #self.emebed = nn.Linear(89,self.atom_feat_num*self.nheads,bias =False)
        self.fc = nn.Linear(self.atom_feat_num*self.nheads*6,args.hidden_size,bias = True)
        self.sum=0
        self.use_att = False
        #self.act = nn.LeakyReLU(0.2)
        #self.atc = nn.ReLU()
        #self.act= nn.functional.relu
        self.act= nn.functional.elu
        
        
        self.SelfAttention = SelfAttention(self.atom_feat_num*self.nheads,8,dropout_prob = 0.0)


        self.gat = GATConv(89,self.atom_feat_num,self.nheads,
                    feat_drop = 0.0,
                    attn_drop = 0.,
                    mode = args.mode,
                    att = args.att,
                    residual = False,
                    #activation= nn.functional.elu,
                   )
        """
        self.gat2 = GATConv(self.atom_feat_num*self.nheads,
                    self.atom_feat_num*self.nheads,1,
                    feat_drop=0.,attn_drop=0.,
                    mode = args.mode,
                    att = args.att,               
                    residual = True,
                    #activation= nn.functional.elu,
                    activation= nn.functional.relu,
                    )
        """
        
        self.gat3 = GATConv(self.atom_feat_num*self.nheads,
                    self.atom_feat_num*self.nheads,1,
                    feat_drop=0.,attn_drop=0.,
                    mode = args.mode,
                    att = args.att,               
                    residual = True,
                    activation= nn.functional.elu,
                    #activation= nn.functional.relu,
                    )
          
        self.mpnn3 = MPNN(self.atom_feat_num*self.nheads,
                   self.atom_feat_num*self.nheads,
                   )

        self.mpnn4 = MPNN(self.atom_feat_num*self.nheads,
                   self.atom_feat_num*self.nheads,
                   )
        #self.mpnn5 = MPNN(self.atom_feat_num*self.nheads,
         #          self.atom_feat_num*self.nheads,
        #           )

    def get_bond_feat(self,mol,graph,distmatrix):
        e = graph.number_of_edges()

        src,dst = graph.edges()
        edge_feats = []
        bond_type = {"SINGLE":1,"DOUBLE":2,"TRIPLE":3,"AROMATIC":4}
        bond_stereo = {"STEREONONE":0, "STEREOANY":1, "STEREOZ":2, "STEREOE":3}
        for index in range(e):
            a = int(src[index])
            b = int(dst[index])
            bond_feat = [0]*8
            if a == b :
                bond_feat[0]=1
            else:
                bond = mol.GetBondBetweenAtoms(a,b)
                btype = str(bond.GetBondType())
                #print(btype)
                idx =  1 if btype not in bond_type else bond_type[btype]
                bond_feat[idx] = 1

                dist = distmatrix[a][b]
                bond_feat[5] = (dist-1)
                

                bond_feat[6] = 1 if bond.IsInRing() else 0
                
                bond_feat[7]=1 if bond.GetIsConjugated() else 0

                #bstereo = str(bond.GetStereo())
                #idx = 8+bond_stereo[bstereo]
                #bond_feat[idx] = 1
                #bond_feat[8] = 1/dist
                #bond_feat[9] = 1/(dist*dist)
                #bond_feat[10] = bond_feat[9]*bond_feat[8]
                #
                

            edge_feats.append(bond_feat)

        return edge_feats

    def k_hop_g(self,_diste,_dist3d,k):
        diste = copy.deepcopy(_diste)
        #print("compute hop",k)
        #print("diste",diste)
        dist3d = copy.deepcopy(_dist3d)
        diste[diste!=k]=0
        diste[diste>0]=1
        #print(diste)
        #diste[np.diag_indices_from(diste)] = 1
        dist = diste*dist3d*dist3d
        #print("dist matrix",dist)
        #print(dist)
        distweight = scipy.sparse.coo_matrix(dist)
        distg = dgl.from_scipy(distweight,eweight_name="dist")
        #didtg = dgl.remove_self_loop(distg)
        #print(distg)
        return distg

    def forward2(self,smiles):
        #smiles = ["Cc1cc(cc(c1)O)C"]
        
        mols = create_graph(smiles, self.args)
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()
        graphs = []

        dist3graphs = []
        dist4graphs = []


        for i,one in enumerate(smiles):
            if one in self.smile2graph:
                g = self.smile2graph[one]

                graph3 = self.smile2dist3[one]
                graph4 = self.smile2dist4[one]


            else:
                self.sum+=1
                if self.sum%100 ==0:
                    print("proceeded",self.sum)
                mol = Chem.MolFromSmiles(one)
                #mol = Chem.AddHs(mol)
                adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
                adj[np.diag_indices_from(adj)] = 1
                cooadj = scipy.sparse.coo_matrix(adj)
                diste = Chem.GetDistanceMatrix(mol)
                dist3d = Chem.GetDistanceMatrix(mol)
                """

                if AllChem.EmbedMolecule(mol) == -1:
                    Chem.rdmolops.RemoveStereochemistry(mol)
                    if AllChem.EmbedMolecule(mol,useRandomCoords=True,maxAttempts=100) == -1 :
                        dist3d = Chem.GetDistanceMatrix(mol)*1.25
                        print("emebed failed,*****************************")
                    else:
                        dist3d=AllChem.Get3DDistanceMatrix(mol)
                else:
                    dist3d=AllChem.Get3DDistanceMatrix(mol)
                """

                g = dgl.from_scipy(cooadj)
                edge_feat = self.get_bond_feat(mol,g,dist3d)

                edge_feat_tensor = torch.Tensor(edge_feat)
                g.edata['bond'] = edge_feat_tensor
                self.smile2graph[one] = g


                graph3 = self.k_hop_g(diste,dist3d,3)
                self.smile2dist3[one] = graph3

                graph4 = self.k_hop_g(diste,dist3d,4)
                self.smile2dist4[one] = graph4



            graphs.append(g)

            dist3graphs.append(graph3)
            dist4graphs.append(graph4)


       
        biggraph = dgl.batch(graphs).to("cuda")
   
        biggraph3 = dgl.to_float(dgl.batch(dist3graphs)).to("cuda")
        biggraph4 = dgl.to_float(dgl.batch(dist4graphs)).to("cuda")


        batch_feature = self.gat(biggraph,atom_feature[1:]).view(-1,self.atom_feat_num*self.nheads)
        
        biggraph.ndata['gat_feat']=self.gat2(biggraph,batch_feature).view(-1,self.atom_feat_num*self.nheads)



        distfeat = biggraph.ndata['gat_feat']
        #distfeat = nn.functional.relu(distfeat)

        distfeat = distfeat + self.mpnn3(biggraph3,distfeat)
        distfeat = nn.functional.relu(distfeat)

        distfeat= distfeat + self.mpnn4(biggraph4,distfeat)
        distfeat = nn.functional.elu(distfeat)

        biggraph.ndata["mpnn_feat"] = distfeat

        #ft = self.dmp

        gnn_out =torch.cat([
                           dgl.sum_nodes(biggraph, 'gat_feat'),
                           dgl.max_nodes(biggraph, 'gat_feat'),
                           dgl.mean_nodes(biggraph, 'gat_feat'),

                           dgl.sum_nodes(biggraph, 'mpnn_feat'),
                           dgl.max_nodes(biggraph, 'mpnn_feat'),
                           dgl.mean_nodes(biggraph, 'mpnn_feat'),
                           ],dim=-1)

        
        
        gnn_out = gnn_out.view(-1,6,self.atom_feat_num*self.nheads)
        gnn_out = self.SelfAttention(gnn_out)
        gnn_out = gnn_out.view(-1,self.atom_feat_num*self.nheads*6)
        return self.fc(gnn_out)

    def forward(self,smiles):
        if self.args.deep:
            return self.forward2(smiles)

        #smiles = ["Cc1cc(cc(c1)O)C"]
        mols = create_graph(smiles, self.args)
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()
        graphs = []
        distgraphs=[]

        for i,one in enumerate(smiles):
            if one in self.smile2graph:
                g = self.smile2graph[one]
                distg = self.smile2dist[one]
            else:
                self.sum+=1
                if self.sum%100 ==0:
                    print("proceeded",self.sum)
                mol = Chem.MolFromSmiles(one)
                #mol = Chem.AddHs(mol)
                adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
                adj[np.diag_indices_from(adj)] = 1
                cooadj = scipy.sparse.coo_matrix(adj)
                self.smile2graph[one] = cooadj

                if AllChem.EmbedMolecule(mol) == -1:
                    Chem.rdmolops.RemoveStereochemistry(mol)
                    if AllChem.EmbedMolecule(mol,useRandomCoords=True,maxAttempts=10) == -1 :
                        dist = Chem.GetDistanceMatrix(mol)*1.25
                        print("emebed failed,*****************************")
                    else:
                        dist=AllChem.Get3DDistanceMatrix(mol)
                else:
                    dist=AllChem.Get3DDistanceMatrix(mol)

                g = dgl.from_scipy(cooadj)
                edge_feat = self.get_bond_feat(mol,g,dist)
                
                #self.smile2edge[one]=edge_feat

                edge_feat_tensor = torch.Tensor(edge_feat)
                #print(edge_feat)
                #print(edge_feat_tensor.size())
                g.edata['bond'] = edge_feat_tensor
                self.smile2graph[one] = g
                """
                diste1 = Chem.GetDistanceMatrix(mol)
                diste1[diste1>4] = 0
                diste1[diste1<3] = 0
                #diste1[(diste1!=3 and diste1!=4)] = 0
                diste1 = diste1*diste1
                distweight = scipy.sparse.coo_matrix(diste1)
                distg = dgl.from_scipy(distweight,eweight_name="dist")
                self.smile2dist[one]=distg
                """
                


                
                
                hop2 = adj
                hop2=np.linalg.matrix_power(adj,2)
                hop2[hop2>0]=1
                hopk=1-hop2
                dist[dist>5.]=0
                dist = hopk*dist*dist
                #print(dist)
                distweight = scipy.sparse.coo_matrix(dist)
                distg = dgl.from_scipy(distweight,eweight_name="dist")
                self.smile2dist[one]=distg
                

                

            graphs.append(g)
            distgraphs.append(distg)
       
        biggraph = dgl.batch(graphs).to("cuda")
        distgraph = dgl.batch(distgraphs).to("cuda")
        distgraph = dgl.to_float(distgraph)

        feat = self.gat(biggraph,atom_feature[1:])
        feat = feat.view(-1,self.atom_feat_num*self.nheads)

        #feat = self.gat2(biggraph,feat)
        #feat = feat.view(-1,self.atom_feat_num*self.nheads)

        feat = self.gat3(biggraph,feat)
        feat = feat.view(-1,self.atom_feat_num*self.nheads)

        biggraph.ndata['gat_feat']=feat
        #biggraph.ndata['gat_feat'] = batch_feature

        #gat_outs = torch.stack(gat_outs, dim=0)
        distfeat = biggraph.ndata['gat_feat']
        distfeat = self.mpnn3(distgraph,distfeat)+distfeat
        distfeat =nn.functional.relu(distfeat)
        
        distfeat = self.mpnn4(distgraph,distfeat)+distfeat
        #distfeat = nn.functional.relu(distfeat)
        
        
        #distfeat = self.mpnn5(distgraph,distfeat)+distfeat
        distfeat = nn.functional.elu(distfeat)

        distgraph.ndata['mpnn_feat'] = distfeat
        '''
        args.init_lr = 1e-5
           args.max_lr = 6e-4
          #args.max_lr = 6e-4
          args.final_lr = 1e-5
        args.warmup_epochs = 3.0
         args.num_lrs = 1
        '''
        
        gnn_out =torch.cat([
                           dgl.sum_nodes(biggraph, 'gat_feat'),
                           dgl.max_nodes(biggraph, 'gat_feat'),
                           dgl.mean_nodes(biggraph, 'gat_feat'),

                           dgl.sum_nodes(distgraph, 'mpnn_feat'),
                           dgl.max_nodes(distgraph, 'mpnn_feat'),
                           dgl.mean_nodes(distgraph, 'mpnn_feat'),
                           
                           ],dim=-1)
           


        """gnn_out = torch.stack([
                           dgl.sum_nodes(biggraph, 'gat_feat'),
                           dgl.max_nodes(biggraph, 'gat_feat'),
                           dgl.sum_nodes(distgraph, 'mpnn_feat'),
                           dgl.max_nodes(distgraph, 'mpnn_feat'),
                           dgl.mean_nodes(distgraph, 'mpnn_feat'),
                           dgl.mean_nodes(biggraph, 'gat_feat')
                           ],dim=1)
        """
        
        gnn_out = gnn_out.view(-1,6,self.atom_feat_num*self.nheads)
        #gnn_out = self.SelfAttention(gnn_out)#.view(-1,self.atom_feat_num*self.nheads*6)
        #gnn_out = self.SelfAttention3(gnn_out)
        gnn_out = self.SelfAttention(gnn_out).view(-1,self.atom_feat_num*self.nheads*6)
        #gnn_out =self.att(gnn_out)
        gnn_out = gnn_out.view(-1,self.atom_feat_num*self.nheads*6)
        
        
        return self.fc(gnn_out)

class FpgnnModel(nn.Module):
    def __init__(self,is_classif,gat_scale,cuda,dropout_fpn):
        super(FpgnnModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self,args):
        self.gnn = GNN(args)
    
    def create_fpn(self,args):

        self.encoder2 = FPAN(args)

    
    def create_scale(self,args):
        #print("createscale")
        linear_dim = int(args.hidden_size)
        #print("linear_dim:",linear_dim)
        
        if self.gat_scale == 0:
            self.fc_fpn = nn.Linear(linear_dim,linear_dim)

    def create_ffn(self,args):
        linear_dim = args.hidden_size
        hid_size = linear_dim
        self.ffn = nn.Sequential(
                                nn.Dropout(self.dropout_fpn),
                                nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                nn.ReLU(),
                                nn.Dropout(self.dropout_fpn),
                                nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                )       

    def forward(self,input):
        if self.gat_scale == 1:
            output = self.gnn(input)
        elif self.gat_scale == 0:
            output = self.encoder2(input)
        else:
            gat_out = self.encoder3(input)
            fpn_out = self.encoder2(input)
            #print("gatoutsize",gat_out.size())
            #gat_out = self.fc_gat(gat_out)
            gat_out = self.act_func(gat_out)
            #print("gatfuturesize",gat_out.size())
            fpn_out = self.fc_fpn(fpn_out)
            fpn_out = self.act_func(fpn_out)
            
            output = torch.cat([gat_out,fpn_out],axis=1)
            #print(output.size())
        output = self.ffn(output)
        
        if self.is_classif and not self.training:
            output = self.sigmoid(output)
        
        return output

def get_atts_out():
    return atts_out

def FPGNN(args):
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = FpgnnModel(is_classif,args.gat_scale,args.cuda,args.dropout)
    #print("gatscale",args.gat_scale)
    if args.gat_scale == 1:
        model.create_scale(args)
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_fpn(args)
        model.create_ffn(args)
    else:
        model.create_gat(args)
        model.create_fpn(args)
        model.create_scale(args)
        model.create_ffn(args)
    
    for param in model.parameters():
        if param.dim() == 1:
            #torch.nn.init.normal_(param)
            nn.init.constant_(param, 0)
        else:
            #torch.nn.init.xavier_uniform_(param, gain=1.0)
            nn.init.xavier_normal_(param)
        
    
    return model

