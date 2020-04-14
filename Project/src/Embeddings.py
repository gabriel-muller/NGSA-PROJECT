import igraph
import csv
from Deepwalk import Deepwalk 
from DeepwalkWithRegularisation import DeepwalkWithRegularisation
from GEMSEC import GEMSEC 
from GEMSECWithRegularisation import GEMSECWithRegularisation

'''Parameters'''

path_inputs = '..\data'
path_outputs = '..\outputs'
filename = '\\TVshow_edges.csv'


'''Graph creation'''

def create_graph(path, filename):
    with open(path + filename, "r") as f:
        reader = csv.reader(f)
        edges  = list(reader)[1:]
    edges = [(int(edge[0]),int(edge[1])) for edge in edges]
    nb_nodes = max([max(nodes) for nodes in edges])+1
    g = igraph.Graph()
    g.add_vertices(nb_nodes)
    g.add_edges(edges)
    return g

'''Model creation'''

def main(model_name = "GEMSEC", walk_number=5, walk_length=80, dimensions=16, 
        negative_samples=10, window_size=5 , clusters=20, gamma_initial=0.1, gamma_final = 0.5,
         learning_rate_initial = 0.01, learning_rate_final = 0.001, Lambda = 2.0**-4):
    if model_name == "Deepwalk":
        model = Deepwalk(walk_number, walk_length, dimensions, negative_samples, window_size, 
                         learning_rate_initial, learning_rate_final)
    if model_name == "DeepwalkWithRegularisation":
        model = DeepwalkWithRegularisation(walk_number, walk_length, dimensions, negative_samples, window_size ,
                                           learning_rate_initial, learning_rate_final, Lambda)
    if model_name == "GEMSEC":
        model = GEMSEC(walk_number, walk_length, dimensions, negative_samples, window_size ,
                       clusters, gamma_initial, gamma_final, learning_rate_initial, learning_rate_final)
    if model_name == "GEMSECWithRegularisation":
        model = GEMSECWithRegularisation(walk_number, walk_length, dimensions, negative_samples, window_size ,
                                         clusters, gamma_initial, gamma_final, learning_rate_initial, learning_rate_final, Lambda)
    return model
        

if __name__ == "__main__":
    graph = create_graph(path_inputs, filename)
    model = main()
    model.fit(graph)
    print("Modularity {}".format(model.get_modularity(graph)))
    model.save_results(path_outputs)
    
    