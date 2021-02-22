import os
import pickle5 as pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ControllerPlacement_env as game
import Node as nd
import numpy as np
import MCTS
import random
import time

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


def generateAlternateGraph(num_clusters: int, num_nodes: int, weight_low: int = 0, weight_high: int = 100,
                           draw=True) -> (nx.Graph, list, dict):
    """
    Generates graph given number of clusters and nodes
    Args:
        num_clusters: Number of clusters
        num_nodes: Number of nodes
        weight_low: Lowest possible weight for edge in graph
        weight_high: Highest possible weight for edge in graph
        draw: Whether or not to show graph (True indicates to show)

    Returns:
        Graph with nodes in clusters, array of clusters, graph position for drawing
    """
    node_colors = np.arange(0, num_nodes, 1, np.uint8)  # Stores color of nodes
    total_nodes = 0
    remainder = num_nodes % num_clusters
    clusters = []  # Stores nodes in each cluster
    # organize number of nodes per cluster and assign node colors
    temp = 0
    # fill in cluster and temp cluster variables and set up node_colors variable
    for x in range(num_clusters):
        if remainder > x:
            nodes_per_cluster = int(num_nodes / num_clusters) + 1
        else:
            nodes_per_cluster = int(num_nodes / num_clusters)

        node_colors[temp + np.arange(nodes_per_cluster)] = x
        temp += nodes_per_cluster
        clusters.append(list(np.arange(nodes_per_cluster) + total_nodes))
        total_nodes += nodes_per_cluster
    G = nx.Graph()
    cluster_endpoints = []

    # create first cluster
    cluster = nx.full_rary_tree(int(np.log2(len(clusters[0]))), len(clusters[0]))

    temp = 0  # variable used to ensure diameter is as small as possible
    while nx.diameter(cluster) > (np.log2(len(clusters[0])) + temp):
        cluster = nx.full_rary_tree(int(np.log2(len(clusters[0]))), len(clusters[0]))
        temp += 1
    nx.set_node_attributes(cluster, 0, 'cluster')

    # set initial edge weight of first cluster
    for (u, v) in cluster.edges():
        cluster.edges[u, v]['weight'] = np.random.random() * 0.75 * (
                    weight_high - weight_low) + weight_low + 0.25 * (weight_high - weight_low)

    inner_cluster_edges = np.random.randint(0, len(clusters[0]),
                                            (int(np.log2(len(clusters[0]))), 2))

    # add edge weights to new edges of first cluster
    inner_cluster_edges = [(u, v, np.random.random() * 0.75 * (weight_high - weight_low) + weight_low + 0.25 * (
                weight_high - weight_low)) for u, v in inner_cluster_edges]
    cluster.add_weighted_edges_from(inner_cluster_edges)

    G = nx.disjoint_union(G, cluster)

    # create other clusters
    for i in range(1, num_clusters):
        # create cluster
        cluster = nx.full_rary_tree(int(np.log2(len(clusters[i]))), len(clusters[i]))
        temp = 0
        while nx.diameter(cluster) > (np.log2(len(clusters[i])) + temp):
            cluster = nx.full_rary_tree(int(np.log2(len(clusters[i]))), len(clusters[i]))
            temp += 1

        nx.set_node_attributes(cluster, i, 'cluster')

        # set initial edge weights
        for (u, v) in cluster.edges():
            if not (u in clusters[x][:len(clusters[x]) // 2]) or v in clusters[x][:len(clusters[x]) // 2]:
                cluster.edges[u, v]['weight'] = np.random.random() * 0.20 * (
                            weight_high - weight_low) + weight_low + 0.05 * (weight_high - weight_low)
            else:
                cluster.edges[u, v]['weight'] = np.random.random() * 0.05 * (weight_high - weight_low) + weight_low

        G = nx.disjoint_union(G, cluster)

        # add connections from new clusters to first cluster
        cluster_endpoint = np.random.randint(0, len(clusters[0]))
        cluster_endpoints.append(cluster_endpoint)
        G.add_edge(cluster_endpoint, np.random.choice(clusters[i][(len(clusters[i]) // 2):]),
                   weight=np.random.random() * 0.20 * (weight_high - weight_low) + weight_low + 0.05 * (
                               weight_high - weight_low))

    # adding inter and inner edges of the clusters
    closest_length = 1000
    nearest_cluster = 0
    shortest_path = 0
    for i in range(1, num_clusters):
        # check for closest cluster besides main cluster
        for x in range(2, num_clusters - 1):
            shortest_path = nx.shortest_path_length(G, cluster_endpoints[i - 1], cluster_endpoints[x - 1])
            if shortest_path < closest_length:
                closest_length = shortest_path
                nearest_cluster = x

        # add inner_cluster_edges
        # get two random points inside a cluster
        inner_cluster_edges = np.random.randint(clusters[i][0], clusters[i][-1] + 1,
                                                (int(np.log2(len(clusters[i]))), 2))
        inner_cluster_edges = [(u, v, np.random.random() * 0.05 * (weight_high - weight_low) + weight_low) for
                               u, v in inner_cluster_edges]
        # cluster.add_weighted_edges_from(inner_cluster_edges)
        G.add_weighted_edges_from(inner_cluster_edges)

        # if the nearest_cluster is too far away, don't add inter-cluster edges
        if shortest_path > (np.random.randint(np.log2(len(clusters[i])), np.log2(len(clusters[i])) + 1)):
            continue

        # add inter_cluster_edges
        inter_cluster_edges = np.random.randint(clusters[i][len(clusters[i]) // 2], clusters[i][-1] + 1,
                                                (int(len(clusters[i]) / (
                                                        np.random.randint(0, (np.log2(len(clusters[i])))) + 1))))
        inter_cluster_edges = [[y, np.random.randint(clusters[nearest_cluster][len(clusters[i]) // 2],
                                                     clusters[nearest_cluster][-1] + 1),
                                np.random.random() * 0.20 * (weight_high - weight_low) + weight_low + 0.05 * (
                                            weight_high - weight_low)] for y in
                               inter_cluster_edges]

        # cluster.add_weighted_edges_from(inner_cluster_edges)
        G.add_weighted_edges_from(inter_cluster_edges)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops caused by adding random edge

    pos = nx.spring_layout(G)

    # Draw graph if True
    if False:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos)
        nx.draw_networkx_edges(G, pos, G.edges())
        plt.draw()
        plt.show()

    return G, clusters, pos







if __name__ == "__main__":

    start_time = time.time()
    finishtime1 = 0

    graph = None
    clusters = None
    pos = None


    # These are your two main parameters which determine what size graph is generated.
    # In order to generate a new graph delete one of the pickle files

    NUMBEROFNODES = 200
    NUMBEROFCLUSTERS = 5



    if os.path.isfile('clusters.pickle') and os.path.isfile('graph.gpickle') and os.path.isfile('position.pickle'):
        print("Found graph from file, using saved graph")
        clusters = pickle.load(open('clusters.pickle', 'rb'))
        pos = pickle.load(open('position.pickle', 'rb'))
        graph = nx.read_gpickle('graph.gpickle')

    else:
        print("Generating graph")
        start_time = time.time()



        graph, clusters, pos = generateAlternateGraph(NUMBEROFCLUSTERS,NUMBEROFNODES)

        nx.write_gpickle(graph, 'graph.gpickle')
        pickle.dump(clusters, open('clusters.pickle', 'wb'))
        pickle.dump(pos, open('position.pickle', 'wb'))
        finishtime1 = time.time() - start_time

    print("--- %s seconds ---" % (time.time() - start_time))
    print("Generated graph")


    # Number of times that MCTS is run for a given graph on different seeds
    test_iterations = 15

    max_score = -1000000
    max_controllers = []

    min_score = 1000000
    min_controllers = []

    score_controllers = []


    for i in range(test_iterations):
        print("Begin test: "+str(i))
        iter_time = time.time()
        np.random.seed(i+65)


        # Set to true to see every iteation of a single MCTS test
        prints = False # print each iteration

        # Generating Environment for test
        RootState = game.State(clusters)
        Root = nd.Node(RootState)
        start_time = time.time()
        environment = game.ControllerPlacement_env(Root, graph, prints)

        # Running MCTS Test with generated environment
        x = MCTS.MCTS(environment, False, prints)
        x.Run()

        # Tracking results of MCTS test run
        score_controllers.append([x.maxScore,x.maxControllers,"--- %s Runtime seconds ---" % (time.time() - iter_time)])
        print([x.maxScore,x.maxControllers,"--- %s Runtime seconds ---" % (time.time() - iter_time)])

        # Tracking best score
        if x.maxScore > max_score:
            max_score = x.maxScore
            max_controllers = x.maxControllers

        if x.maxScore < min_score:
            min_score = x.maxScore
            min_controllers = x.maxControllers
        print("--- %s Runtime seconds ---" % (time.time() - iter_time))

    sum = 0
    for i in range(test_iterations):
        print(score_controllers[i])
        sum = sum + score_controllers[i][0]

    # Print total results
    print("Average: "+str(sum/test_iterations))
    print("Best Score: "+str(max_score)+"---"+str(max_controllers))
    print("Worst Score: " + str(min_score) + "---" + str(min_controllers))



    # y_list = []
    #
    # # print(self.calculateOptimal())
    # with open('Results.txt') as f:
    #     for line in f:
    #         num = line.split()[0]
    #         y_list.append(float(num))
    #
    # plt.plot([i for i in range(len(y_list))], y_list)
    # plt.title('Max Score Vs Iteration Step')
    # plt.xlabel('Iteration Step')
    # plt.ylabel('Max Score')
    # plt.show()