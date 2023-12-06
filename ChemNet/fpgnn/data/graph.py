from argparse import Namespace
from rdkit import Chem
import torch
import os
import pandas as pd
import pickle

atom_type_max = 100
atom_f_dim = 89
atom_features_define = {
    #'atom_symbol': list(range(atom_type_max)),
    'atom_symbol':[1,3,5,6,7,8,9,11,12,13,14,15,16,17,19,20,22,23,24,25,26,27,28,29,\
            30,31,32,33,34,35,40,42,44,45,46,47,50,51,52,53,55,64,65,67,74,75,77,78,79,80,\
            81,82,83,89,92
    ],
    "radical_electrons":[0,1,2],
    'degree': [0, 1, 2, 3, 4, 5,6],
    'formal_charge': [-1, -2, 1, 2, 0,-3,3],
    'charity_type': [0, 1, 2, 3],
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        0,
        1,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],}

smile_changed = {}

smile2atomfeat = {}

def get_atom_features_dim():
    return atom_f_dim

def onek_encoding_unk(key,length,encode_unknow = False):
    #encoding = [0] * (len(length) + 1)
    #index = length.index(key) if key in length else -1
    #print(length)

    encoding = [0] * len(length)
    if key not in length:
        index = -1
    else:
        index = length.index(key)
    encoding[index] = 1

    return encoding

def get_atom_feature(atom):
    feature = onek_encoding_unk(atom.GetAtomicNum(), atom_features_define['atom_symbol']) + \
           onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
           onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + [atom.GetMass() * 0.01]

    return feature

class GraphOne:
    def __init__(self,smile,args):
        self.smile = smile
        self.atom_feature = []
        donorsmarts = '[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),$([$(n[n;H1]),$(nc[n;H1])])]),$([O,S;H1;+0])]'

        acceptorsmarts = "[$([O;H1;v2]),$([O;H0;v2;!$(O=N-*),$([O;-;!$(*-N=O)]),$([o;+0])]),$([n;+0;!X3;!$([n;H1](cc)cc),$([$([N;H0]#[C&v4])]),$([N&v3;H0;$(Nc)])]),$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]"

        donor = Chem.MolFromSmarts(donorsmarts)
        acceptor = Chem.MolFromSmarts(acceptorsmarts)
        
        mol = Chem.MolFromSmiles(self.smile)
        #mol = Chem.AddHs(mol)

        donor_match = sum(mol.GetSubstructMatches(donor), ())
        acceptor_match = sum(mol.GetSubstructMatches(acceptor), ())
        self.atom_num = mol.GetNumAtoms()
        
        for i, atom in enumerate(mol.GetAtoms()):
            Hfeat = [1 if i in donor_match else 0]+[1 if i in acceptor_match else 0]
            self.atom_feature.append(get_atom_feature(atom)+Hfeat)
        self.atom_feature = [self.atom_feature[i] for i in range(self.atom_num)]
        self.atom_feature = torch.Tensor(self.atom_feature)
        
        
class GraphBatch:
    def __init__(self,graphs,args,total_nodes):
        #smile_list = []
        #for graph in graphs:
            #smile_list.append(graph.smile)
        #self.smile_list = smile_list
        #self.smile_num = len(self.smile_list)

        self.atom_feature_dim = get_atom_features_dim()
        self.atom_no = 1
        self.atom_index = []

        #atom_feature = [[0]*self.atom_feature_dim]
        self.atom_feature = torch.empty((total_nodes,self.atom_feature_dim))
        for graph in graphs:
            atom_size = graph.atom_num
            self.atom_feature[self.atom_no:self.atom_no+atom_size] = graph.atom_feature
            #atom_feature.extend(graph.atom_feature)
            self.atom_index.append((self.atom_no,graph.atom_num))
            self.atom_no += atom_size

        #self.atom_feature = torch.FloatTensor(atom_feature) 

    def get_feature(self):
        return self.atom_feature,self.atom_index

def create_graph(smile,args):
    graphs = []
    total_nodes = 1
    for one in smile:
        #graph = GraphOne(one, args)
        #print(smile_changed)
        #print("okokok")
        #smile_changed[one] = graph

        if one in smile_changed:
            graph = smile_changed[one]
        else:
            graph = GraphOne(one, args)
            smile_changed[one] = graph
            
        total_nodes += graph.atom_num
        graphs.append(graph)
    return GraphBatch(graphs,args,total_nodes)

def prepare_feature(args):
    

    data_path = args.data_path
    dataset = data_path.split("/")[-1][:-4]
    path = os.path.join("data","{}.feature".format(dataset))

    if not os.path.exists(path):
        df = pd.read_csv(data_path)
        smiles = df.iloc[1:,0]
        for smile in smiles:
            graph = GraphOne(smile, args)
            smile_changed[smile] = graph
        #create_graph(smiles,args)
        print(smile_changed)
        
        with open(path,"wb") as f:
            pickle.dump(smile_changed,f)
    else:
        with open(path,"rb") as f:
            smile_changed = pickle.load(f)

