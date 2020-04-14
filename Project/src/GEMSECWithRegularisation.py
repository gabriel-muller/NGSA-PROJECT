import numpy as np
import random
import json
import pandas as pd
from tqdm import tqdm


class GEMSECWithRegularisation(object):


    def __init__(self, walk_number=5, walk_length=80, dimensions=16, negative_samples=10,
                 window_size=5 , clusters=20, gamma_initial=0.1, gamma_final = 0.5,
                 learning_rate_initial = 0.01, learning_rate_final = 0.001, Lambda = 2.0**-4):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.negative_samples = negative_samples
        self.window_size = window_size
        self.clusters = clusters
        self.walker = []
        self.gamma_initial = gamma_initial
        self.current_gamma = gamma_initial
        self.gamma_final = gamma_final
        self.learning_rate_initial = learning_rate_initial
        self.current_learning_rate = learning_rate_initial
        self.learning_rate_final = learning_rate_final
        self.Lambda = Lambda
        
    def _setup_sampling_weights(self, graph):

        self.sampler = {}
        index = 0
        for node in graph.vs():
            for _ in range(node.degree()):
                self.sampler[index] = node.index
                index = index + 1
        self.global_index = index-1


    def _initialize_node_embeddings(self):

        shape = (self.nb_nodes, self.dimensions)
        self._base_embedding = np.random.normal(0, 1.0/self.dimensions, shape)

        
    def _initialize_cluster_centers(self):

        shape = (self.dimensions, self.clusters)
        self._cluster_centers = np.random.normal(0, 1.0/self.dimensions, shape)
        
        
    def _gamma_incrementer(self, step):
        
        if step > 1 and step <= self.num_steps:
            exponent = (-np.log10(self.gamma_initial))/self.num_steps
            self.current_gamma = self.current_gamma * (10 **exponent)*(self.gamma_final-self.gamma_initial)
            self.current_gamma = self.current_gamma + self.gamma_initial
        
        
    def _learning_rate_incrementer(self, step):
        if step > 1 and step <= self.num_steps:
            self.current_learning_rate = (self.learning_rate_initial - self.learning_rate_final) * (1 - step/self.num_steps)
    
    
    def _random_walks(self, graph):
    
        nodes = [node.index for node in graph.vs]
        for rep in range(0, self.walk_number):
            random.shuffle(nodes)
            print("Random walk series " + str(rep+1))
            for node in tqdm(nodes):
                walk = graph.random_walk(node, self.walk_length)
                self.walker.append(walk)
    
    
    def _sample_negative_samples(self):

        negative_samples = [self.sampler[random.randint(0,self.global_index)] for _ in range(self.negative_samples)]
        return negative_samples


    def _calculcate_noise_vector(self, negative_samples, source_node):

        noise_vectors = self._base_embedding[negative_samples, :]
        source_vector = self._base_embedding[int(source_node), :]
        raw_scores = noise_vectors.dot(source_vector.T)
        raw_scores = np.exp(np.clip(raw_scores, -15, 15))
        scores = raw_scores/np.sum(raw_scores)
        scores = scores.reshape(-1,1)
        noise_vector = np.sum(scores*noise_vectors,axis=0)
        return noise_vector


    def _calculate_cluster_vector(self, source_node):

        distances = self._base_embedding[int(source_node), :].reshape(-1,1) - self._cluster_centers
        scores = np.power(np.sum(np.power(distances,2),axis=0),0.5)
        cluster_index = np.argmin(scores)
        cluster_vector = distances[:,cluster_index]/scores[cluster_index]
        return cluster_vector, cluster_index
    
    def _regularisation(self,graph, source_node, target_node):
        
        weight = graph.similarity_jaccard(pairs = [(source_node,target_node)])[0]
        distance = self._base_embedding[int(source_node), :].reshape(-1,1) - self._base_embedding[int(target_node), :].reshape(-1,1)
        score = np.power(np.sum(np.power(distance,2),axis=0),0.5)
        regularisation_vector = distance/score
        return regularisation_vector, weight
    
    
    def _do_descent_for_pair(self, graph, negative_samples, source_node, target_node):

        
        noise_vector = self._calculcate_noise_vector(negative_samples, source_node)
        target_vector = self._base_embedding[int(target_node), :]
        cluster_vector, cluster_index = self._calculate_cluster_vector(source_node)
        regularisation_vector, weight, = self._regularisation(graph, source_node, target_node)
        node_gradient = noise_vector - target_vector + self.current_gamma*cluster_vector + self.Lambda*weight*regularisation_vector
        node_gradient = node_gradient / np.linalg.norm(node_gradient)
        self._base_embedding[int(source_node), :] += -self.current_learning_rate*node_gradient
        self._cluster_centers[:, cluster_index] += self.current_learning_rate*self.current_gamma*cluster_vector 


    def _update_a_weight(self, graph, source_node, target_node):

        
        negative_samples = self._sample_negative_samples()
        self._do_descent_for_pair(graph, negative_samples, source_node, target_node)
        self._do_descent_for_pair(graph, negative_samples, target_node, source_node)


    def _do_gradient_descent(self, graph):

        
        #global_step = 0
        random.shuffle(self.walker)
        for walk in self.walker:
            for i, source_node in enumerate(walk[:self.walk_length-self.window_size]):
                for step in range(1, self.window_size+1):
                    #global_step+=1
                    #self._gamma_incrementer(step)
                    #self._learning_rate_incrementer(step)
                    target_node = walk[i+step]
                    self._update_a_weight(graph, source_node, target_node)


    def fit(self, graph):
        
        self.nb_nodes = len(graph.vs)
        self.num_steps = self.nb_nodes*self.walk_number
        self._setup_sampling_weights(graph)
        self._random_walks(graph)
        self._initialize_node_embeddings()
        self._initialize_cluster_centers()
        self._do_gradient_descent(graph)


    def get_embedding(self):

        return np.array(self._base_embedding)


    def _get_membership(self, node):

        distances = self._base_embedding[node, :].reshape(-1, 1) - self._cluster_centers
        scores = np.power(np.sum(np.power(distances,2), axis=0), 0.5)
        cluster_index = np.argmin(scores)   
        return cluster_index


    def get_memberships(self):

        memberships = {node:self._get_membership(node) for node in range(self._base_embedding.shape[0])}
        return memberships 
    
    def get_modularity(self, graph):
        
        memberships = [self._get_membership(node) for node in range(self._base_embedding.shape[0])]
        return graph.modularity(membership = memberships)
    
    def save_results(self, path):
        
        final_embeddings = pd.DataFrame(self.get_embeddings())
        final_embeddings.to_csv(path + '\GMR_embeddings.csv', index=None)
        c_means = pd.DataFrame(self._cluster_centers)
        c_means.to_csv(path + '\GMR_cluster_means.csv', index=None)
        with open(path+'\GMR_memberships.txt', "w") as outfile:
            json.dump(self.get_memberships(), outfile)
