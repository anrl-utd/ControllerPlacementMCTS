import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Examples import ControllerPlacement_env as game
import Node as nd
import numpy as np
import MCTS
import random
import time
from Examples.ControllerEnv import generateGraph, generateAlternateGraph, generateClusters, ControllerEnv



if __name__ == "__main__":
    graph = None
    clusters = None
    pos = None
    np.random.seed(256)
    # This might be lazy code, but I think it is not worth importing more modules just to check if file exists before trying to open it
    if os.path.isfile('clusters.pickle') and os.path.isfile('graph.gpickle') and os.path.isfile('position.pickle'):
        print("Found graph from file, using saved graph")
        clusters = pickle.load(open('clusters.pickle', 'rb'))
        pos = pickle.load(open('position.pickle', 'rb'))
        graph = nx.read_gpickle('graph.gpickle')
        print(clusters)

    heuristic_env = ControllerEnv(graph, clusters)
    heuristic_controllers, heuristic = heuristic_env.graphCentroidAction()
    print("Heuristic: ")
    print(heuristic_controllers, heuristic)