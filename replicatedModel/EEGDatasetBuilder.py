from itertools import product
import scipy.sparse as sp
import pandas as pd
import numpy as np
from spektral.data import Dataset, Graph




class DatasetBuilder_GCN(Dataset):
    
    def __init__(self, X, weights, Y,indices, **kwargs):
        self.X = X         # (225334, 48)
        self.Y = Y         # (225334,)
        self.weights = weights # (225334,64)
        self.indices = indices

        ch_names = ["F7-F3", "F8-F4", "T7-C3", "T8-C4", "P7-P3", "P8-P4", "O1-P3","O2-P4"]
        self.num_nodes= len(ch_names)
        #self.edge_f= edge_f.reshape(edge_f.shape[0], self.num_nodes*self.num_nodes,1)

        self.rows, self.cols = zip(*product(range(self.num_nodes), repeat=2))
        super().__init__(**kwargs)

        
        
        
    def read(self):
        graphs = []
        for idx in self.indices:
            x = self.X[idx].reshape(self.num_nodes, 6)       # 8 nodos × 6 bandas
            w = self.weights[idx]
            
            adj_matrix_sparse = sp.coo_matrix((w, (self.rows, self.cols)), shape=(self.num_nodes, self.num_nodes))
           
            
            y=self.Y[idx]
            
            graphs.append(Graph(x=x,a=adj_matrix_sparse,y=y))
        return graphs

class DatasetBuilder_noindex(Dataset):
    
    def __init__(self, X, weights, Y, **kwargs):
        self.X = X         # (225334, 48)
        self.Y = Y         # (225334,)
        self.weights = weights # (225334,64)

        ch_names = ["F7-F3", "F8-F4", "T7-C3", "T8-C4", "P7-P3", "P8-P4", "O1-P3","O2-P4"]
        self.num_nodes= len(ch_names)
        #self.edge_f= edge_f.reshape(edge_f.shape[0], self.num_nodes*self.num_nodes,1)

        self.rows, self.cols = zip(*product(range(self.num_nodes), repeat=2))
        super().__init__(**kwargs)

        
        
        
    def read(self):
        graphs = []
        for idx in range(len(self.X)):
            x = self.X[idx].reshape(self.num_nodes, 6)       # 8 nodos × 6 bandas
            w = self.weights[idx]
            
            adj_matrix_sparse = sp.coo_matrix((w, (self.rows, self.cols)), shape=(self.num_nodes, self.num_nodes))
           
            
            y=self.Y[idx]
            
            graphs.append(Graph(x=x,a=adj_matrix_sparse,y=y))
        return graphs
        