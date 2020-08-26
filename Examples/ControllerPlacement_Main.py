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

def generateGraph(num_clusters: int, num_nodes: int, prob_cluster: float = 0.5, prob: float = 0.2, weight_low: int = 0,
                  weight_high: int = 100, draw=True) -> (nx.Graph, list, dict):
    """Generates graph given number of clusters and nodes
    Args:
        num_clusters: Number of clusters
        num_nodes: Number of nodes
        prob_cluster: Probability of adding edge between any two nodes within a cluster
        prob: Probability of adding edge between any two nodes
        weight_low: Lowest possible weight for edge in graph
        weight_high: Highest possible weight for edge in graph
        draw: Whether or not to show graph (True indicates to show)

    Returns:
        Graph with nodes in clusters, array of clusters, graph position for drawing
    """
    node_colors = np.arange(0, num_nodes, 1, np.uint8)  # Stores color of nodes
    G = nx.Graph()
    node_num = 0
    nodes_per_cluster = int(num_nodes / num_clusters)
    clusters = np.zeros((num_clusters, nodes_per_cluster), np.uint8)  # Stores nodes in each cluster

    # Create clusters and add random edges within each cluster before merging them into single graph
    for i in range(num_clusters):
        # Add tree to serve as base of cluster subgraph. Loop through all edges and assign weights to each
        # cluster = nx.random_tree(nodes_per_cluster)
        p = 0.1  # TODO: Move to constants
        cluster = nx.fast_gnp_random_graph(nodes_per_cluster, p)
        while (not nx.is_connected(cluster)):
            cluster = nx.fast_gnp_random_graph(nodes_per_cluster, p)
        for start, end in cluster.edges:
            cluster.add_edge(start, end, weight=50)  # random.randint(weight_low, weight_high))

        # Add edges to increase connectivity of cluster
        # new_edges = np.random.randint(0, nodes_per_cluster, (int(nodes_per_cluster * prob_cluster), 2))
        # new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
        # new_edges = np.append(new_edges, new_weights, 1)
        # cluster.add_weighted_edges_from(new_edges)

        # Set attributes and colors
        nx.set_node_attributes(cluster, i, 'cluster')
        nx.set_node_attributes(cluster, 0.5, 'learning_automation')
        node_colors[node_num:(node_num + nodes_per_cluster)] = i
        node_num += nodes_per_cluster
        clusters[i, :] = np.asarray(cluster.nodes) + nodes_per_cluster * i

        # Merge cluster with main graph
        G = nx.disjoint_union(G, cluster)

    # Add an edge to connect all clusters (to gurantee it is connected)
    node_num = nodes_per_cluster - 1 + nodes_per_cluster
    edge_weight = 1000
    G.add_edge(nodes_per_cluster - 1, nodes_per_cluster, weight=10000)
    for i in range(num_clusters - 1 - 1):
        G.add_edge(node_num, node_num + 1, weight=50)  # random.randint(weight_low, weight_high))
        node_num += nodes_per_cluster
        edge_weight = int(1000 * pow(0.2, i))

    # Add random edges to any nodes to increase diversity
    # new_edges = np.random.randint(0, num_nodes, (int(num_nodes * 0.1), 2))
    # new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
    # new_edges = np.append(new_edges, new_weights, 1)
    # G.add_weighted_edges_from(new_edges)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops caused by adding random edges

    # Draw graph
    pos = nx.spring_layout(G)
    if draw:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, G.edges())
        plt.draw()
        plt.show()
    return G, clusters, pos


def generateClusters(graph: nx.Graph, edge_label: str=None) -> (nx.Graph, list, dict):
    """
    Converts a normal NetworkX graph into a controller-placement graph by adding cluster attributes
    Args:
        graph (nx.Graph): NetworkX graph to convert to controller-placement graph with clusters
        edge_label (str): Optional edge attribute label of original graph to use as edge weights instead of random
    Returns:
        NetworkX graph with 'cluster' node attribute
        List of lists of nodes in clusters
        Graph display rendering position
    """
    # Uses Clauset-Newman-Moore greedy modularity maximization algorithm to partition nodes into communities
    # it does not consider edge weights, sadly
    new_graph = nx.relabel.convert_node_labels_to_integers(graph)  # Converts node IDs to ints in case they weren't before
    clusters = list(nx.algorithms.community.greedy_modularity_communities(new_graph))
    # Add cluster attribute
    node_attrs = {}
    for i in range(len(clusters)):
        node_list = clusters[i]
        for node in node_list:
            node_attrs[node] = {'cluster' : i }
    nx.set_node_attributes(new_graph, node_attrs)

    # Set random edge weights if no edge label is set
    if edge_label is None:
        for (u, v) in new_graph.edges():
            new_graph.edges[u,v]['weight'] = random.randint(0,10)
    else:
        # Use LinkSpeed (unit GB/s) edge attribute as weight
        edge_dict = nx.get_edge_attributes(new_graph, edge_label)
        new_edges = { key: float(value) for key, value in edge_dict.items() }
        nx.set_edge_attributes(new_graph, new_edges, 'weight')

    return new_graph, clusters, nx.kamada_kawai_layout(new_graph)

def generateGraphAlt(num_nodes, num_clusters):
    print("Generating graph")
    # graph, clusters, pos = generateGraph(6, 90, draw=False, weight_low=1, weight_high=10)
    clusters = []
    graph = None
    pos = None
    while len(clusters) < num_clusters:
        k_graph = nx.fast_gnp_random_graph(num_nodes, 0.05)
        while (not nx.is_connected(k_graph)):
            k_graph = nx.fast_gnp_random_graph(num_nodes, 0.05)
        graph, clusters, pos = generateClusters(k_graph)
    nx.write_gpickle(graph, 'graph.gpickle')
    pickle.dump(clusters, open('clusters.pickle', 'wb'))
    pickle.dump(pos, open('position.pickle', 'wb'))


    node_colors = np.arange(0, num_nodes, 1, np.uint8)  # Stores color of nodes
    if True:
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos, graph.edges())
        plt.draw()
        plt.show()
    return graph, clusters, pos



if __name__ == "__main__":
    graph = None
    clusters = None
    pos = None
    # This might be lazy code, but I think it is not worth importing more modules just to check if file exists before trying to open it
    if os.path.isfile('clusters.pickle') and os.path.isfile('graph.gpickle') and os.path.isfile('position.pickle'):
        print("Found graph from file, using saved graph")
        clusters = pickle.load(open('clusters.pickle', 'rb'))
        pos = pickle.load(open('position.pickle', 'rb'))
        graph = nx.read_gpickle('graph.gpickle')

        node_colors = np.arange(0, 100, 1, np.uint8)  # Stores color of nodes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos, graph.edges())
        plt.draw()
        plt.show()
    else:
        print("Generating graph")

        graph, clusters, pos = generateGraphAlt(100,8)

        nx.write_gpickle(graph, 'graph.gpickle')
        pickle.dump(clusters, open('clusters.pickle', 'wb'))
        pickle.dump(pos, open('position.pickle', 'wb'))



    # try:
        # I store the results in a SQLite database so that it can resume from checkpoints.
        # study = optuna.create_study(study_name='ppo_direct', storage='sqlite:///params_select.db', load_if_exists=True)
        # study.optimize(lambda trial: optimize_algorithm(trial, graph, clusters, pos), n_trials=500)

        #train_once(graph, clusters, pos, compute_optimal=False)


    RootState = game.State(clusters)
    Root = nd.Node(RootState)


   #print(game.calculateOptimal(RootState))
    x = MCTS.MCTS(Root, graph, True)
    x.Run()



    # except Exception as e:
    #     print(e)
    #     print('Interrupted, saving . . . ')
    #     nx.write_gpickle(graph, 'graph.gpickle')
    #     pickle.dump(clusters, open('clusters.pickle', 'wb'))
    #     pickle.dump(pos, open('position.pickle', 'wb'))