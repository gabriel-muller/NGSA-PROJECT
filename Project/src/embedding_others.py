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
import re

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

def clean_type(model):
    regex = re.compile("[^a-zA-Z]+")
    name = str(type(model)).split('.')[-1]
    name = regex.sub('', name)
    return name

g_HR, g_dir_HR = create_graph(path,'\HR_edges.csv')
g_HU, g_dir_HU = create_graph(path,'\HU_edges.csv')
g_RO, g_dir_RO = create_graph(path,'\RO_edges.csv')

dc_HR = {key:value for (key, value) in list(zip(list(g_HR.nodes()), [int(n) for n in g_HR.nodes()]))}
dc_HU = {key:value for (key, value) in list(zip(list(g_HU.nodes()), [int(n) for n in g_HU.nodes()]))}
dc_RO = {key:value for (key, value) in list(zip(list(g_RO.nodes()), [int(n) for n in g_RO.nodes()]))}

g_HR = nx.relabel.relabel_nodes(g_HR, dc_HR)
g_HU = nx.relabel.relabel_nodes(g_HU, dc_HU)
g_RO = nx.relabel.relabel_nodes(g_RO, dc_RO)

#%%Modeling

#MNMF
def embed(model, g, cntry):
    model.fit(g)
    embd = model.get_embedding()
    name = clean_type(model)
    np.save(open('../embeddings/' + cntry + '_' + name +'_embedding.npy', 'wb'), embd)
    
cntry = ['HR', 'HU', 'RO']
datasets = [g_HR, g_HU, g_RO]
data = list(zip(cntry, datasets))
models = [kc.MNMF(dimensions=16)] #, kc.Walklets(dimensions=16), kc.DANMF(layers=[32,8]), kc.LaplacianEigenmaps(dimensions=16)#,kc.Walklets(dimensions=16),kc.MNMF(dimensions=16)] 

for cntry, grph in data:
    for model in models:
        t = time.time()
        name = clean_type(model)
        print('##### ' + cntry + ' | ' + name + ' #####')
        embed(model, grph, cntry)
        print('Duration: ' + str(int(time.time() - t)) + ' s')