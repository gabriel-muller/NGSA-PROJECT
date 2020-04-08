# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:14:28 2020

@author: Benjamin Pommier
"""

import karateclub as kc
import numpy as np
import random
import igraph
import networkx as nx
import csv
import json
import pandas as pd
import time
import matplotlib as plt

##Loading data as networkx
path = '..\data'

def create_graph(path, filename):
    with open(path + filename, "r") as f:
        reader = csv.reader(f)
        edges  = list(reader)[1:] 
    g = nx.Graph()
    g_dir = nx.DiGraph()
    g.add_edges_from(edges)
    g_dir.add_edges_from(edges)
    return g, g_dir

g_HR, g_dir_HR = create_graph(path,'\HR_edges.csv')
g_HU, g_dir_HU = create_graph(path,'\HU_edges.csv')
g_RO, g_dir_RO = create_graph(path,'\RO_edges.csv')

#%%Modeling

#MNMF
def embed(model, g, cntry):
    model.fit(g)
    embed = model.get_embedding()
    np.save(open('../output/embedding/' + cntry + 'MNMF_embedding.csv'), embed)
    
cntry = ['HR', 'HU', 'RO']
datasets = [hu, g_HU, g_RO]
data = list(zip(cntry, datasets))
models = [kc.MNMF(dimensions=16), kc.Walklets(dimensions=16)]

for cntry, grph in data:
    for model in models:
        t = time.time()
        print('##### ' + cntry + ' | '+ str(type(model)) + + ' #####')
        print('Duration: ' + str(int(time.time() - t)) + ' s')
        embed(model, grph, cntry)