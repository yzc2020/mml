import time
import os
import copy
import scipy
from scipy import sparse 
import rdkit
from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem, MACCSkeys,ChemicalFeatures
from rdkit import RDConfig


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import deepchem as dc

def bond(path):
    df=pd.read_csv(path)
    SMILES = df['Smiles'].values
    n = len(SMILES)
    print("n",n)
    bondsum=0
    atomsum=0

    for smile in SMILES:
        mol = Chem.MolFromSmiles(smile)
    #获取分子中的原子数目
        mol=Chem.AddHs(mol)
        atom_num = mol.GetNumAtoms()
    #获取分子中的键数目
        bond_num = mol.GetNumBonds()
        bondsum+=bond_num
        atomsum+=atom_num
    
    print("ave atom",1.0*atomsum/n)
    print("ave bond",1.0*bondsum/n)


if __name__ == '__main__':
    #bond("./data/pdbbind_full.csv")
    dir = "./dataset/BreastCellLines"
    filenames = os.listdir(dir)
    for file in filenames:
        path = os.path.join(dir,file)
        print(file)
        bond(path)
    
