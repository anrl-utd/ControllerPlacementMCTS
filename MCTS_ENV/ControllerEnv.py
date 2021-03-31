"""
Environment for reinforcement learning
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings
import random
import itertools
import pprint
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)


class ControllerEnv(gym.Env):
    """Base environment used to simulate the network for the RL"""
    metadata = {'render.modes': ['human']}

    def __init__(self, graph: nx.Graph, clusters: list, pos: dict = None, check_controller_num=True):
        """
        Initializes base controller environment
        Args:
            graph (nx.Graph): NetworkX graph to create controller RL environment for
            clusters (list): List of lists of nodes where ith node list is ith cluster
            pos (dict): Display position for graph when rendering
            check_controller_num (bool): Flag for whether to use asserts to make sure input action has X controllers
        """
        # Define action/observation space for stable_baselines algos
        self.action_space = spaces.Box(np.zeros(len(clusters)), np.ones(len(clusters)) * len(graph.nodes),
                                       dtype=np.uint8)
        # self.observation_space = spaces.Box(np.zeros(shape=len(graph.nodes)), np.ones(shape=len(graph.nodes)), dtype=np.bool)

        self.original_graph = graph.copy()  # Keep original graph in case of needing it for reset
        # Generate graph display positions if needed
        if (pos is None):
            self.pos = nx.kamada_kawai_layout(graph)  # get the positions of the nodes of the graph
        else:
            self.pos = pos
        self.clusters = clusters
        self.graph = graph.copy()
        self.degree = self._graph_degree()
        self.check_controller_num = check_controller_num
        self.current_controllers = None  # Stores controllers placed in last action (used for rendering)
        self.adjacent_clusters = self._get_adjacent_clusters()
        print("graph_files adjacency matrix:")
        print(self.adjacent_clusters)
        print("Initialized environment!")

    def step(self, action: list) -> int:
        """
        Steps the environment once
        Args:
            action (list): List of node indices corresponding to all controllers
        Returns:
            Distance between controllers (total distance in controller metagraph)
        """
        """
        How it works:
        action is indices of controllers
        Create a new complete graph with controllers only
         - Use shortest-distance between controllers as the weight of edges for controller graph
         - Store indices of nodes under each controller
        Create several "packets" with source and destination
        Find controller for source and destination
        If source is same as destination controller, distance is 0
        Otherwise, distance is the shortest-path distance between the source controller and destination
        Add up all distances and have that as initial reward
        """
        controller_graph = None  # Stores controller metagraph
        # Create metagraph of controllers. The node at an index corresponds to the controller of the cluster of that index
        controller_graph, bad_controllers = self._set_controllers(action)
        # Return output reward
        return controller_graph.size(
            weight='weight') + 100000 * bad_controllers  # 100000 distance per invalid controller

    def reset(self):
        """Resets the environment to initial state. Does nothing right now"""
        print("Reset environment")

    def render(self, mode: str = 'human', graph: nx.Graph = None):
        """
        Renders NetworkX graph
        Args:
            mode (str): What mode to render environment in ('human' to display in window)
            graph (nx.Graph): Display another graph besides current environment graph
                                (if I followed SWE standards, this entire function should be outside the environment but eh)
        """
        render_graph = None  # Graph to render
        if (graph is None):
            render_graph = self.graph
        else:
            render_graph = graph

        """Renders the environment once"""
        plt.clf()  # Clear the matplotlib figure

        # Redraw the entire graph (this can only be expedited if we save the position and colors beforehand)
        # Then we won't have to recalculate all this to draw. Maybe make them a global variable?
        node_colors = np.arange(0, nx.number_of_nodes(render_graph), 1)  # Stores colors in every node
        clustering = nx.get_node_attributes(render_graph, 'cluster')
        for node in clustering:
            node_colors[int(node)] = clustering[node]
        node_sizes = np.ones((len(render_graph))) * 300  # 300 is not-controller node size
        # Set controller nodes to larger size
        try:
            if self.current_controllers is not None:
                node_sizes[self.current_controllers] = 1000
        except Exception as e:
            print("Invalid controllers provided to render()")
        # Draw graph
        nx.draw_networkx_nodes(render_graph, self.pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(render_graph, self.pos, render_graph.edges())  # draw the edges of the render_graph
        nx.draw_networkx_labels(render_graph, self.pos)  # draw  the labels of the render_graph
        edge_labels = nx.get_edge_attributes(render_graph, 'weight')
        nx.draw_networkx_edge_labels(render_graph, self.pos,
                                     edge_labels=edge_labels)  # draw the edge weights of the render_graph
        plt.draw()
        if mode is 'human':
            plt.show()
        else:
            plt.savefig(mode,
                        bbox_inches='tight')  # A little hacky, but a way to provide the save file name without creating a parameter

    def _set_controllers(self, controllers: list) -> (nx.Graph, int):
        """
        Creates metagraph of controllers
        Args:
            controllers: Array of controller indices

        Returns:
            Complete graph of controllers (metagraph)

        Raises:
            AssertError: Issue with controller indices (not 1 per cluster)"""
        # Ensure that these are valid controllers - all clusters have a controller
        if self.check_controller_num:
            assert (len(controllers) == len(self.clusters))
        found_clusters = np.zeros((len(self.clusters)))  # Stores what clusters have controllers been found for
        clusters = nx.get_node_attributes(self.graph, 'cluster')
        index = 0
        num_erroneous = 0
        valid_controllers = []
        for controller in controllers:
            # Multiple controllers in a cluster
            # controller = str(controller)  # Just for the GraphML files, remove for generated graphs
            if found_clusters[clusters[controller]] == 0:
                found_clusters[clusters[controller]] = 1
                valid_controllers.append(controller)
            else:
                num_erroneous += 1  # This computation could be removed, but eh I like it [Usaid]

        # Controllers were found to be valid. Now add controllers to metagraph.
        controller_graph = nx.Graph()  # Store controller metagraph
        controller_graph.add_nodes_from(range(len(valid_controllers)))
        # Add edges between controllers in metagraph
        for pair in itertools.combinations(valid_controllers, 2):
            first_cluster = clusters[pair[0]]
            second_cluster = clusters[pair[1]]
            assert first_cluster != second_cluster, "2 controllers in same cluster in _set_controllers {} {}".format(
                first_cluster, second_cluster)
            if self.adjacent_clusters[first_cluster][second_cluster] == 1:
                controller_graph.add_edge(first_cluster, second_cluster,
                                          weight=nx.dijkstra_path_length(self.graph, source=pair[0], target=pair[1]))

        self.current_controllers = valid_controllers.copy()
        return (controller_graph, num_erroneous)

    def _graph_degree(self):
        """Returns the highest degree of a node in the graph"""
        return max([degree for node, degree in self.graph.degree()])

    def _random_valid_controllers(self) -> list:
        """Intended for testing, this gives a random set of valid controllers"""
        cluster_arr = np.asarray(self.graph.nodes.data('cluster'))  # Creates NumPy array with [node #, cluster #] rows
        controller_indices = []
        for cluster in range(len(self.clusters)):  # For every cluster
            cluster_controller = np.random.choice(cluster_arr[cluster_arr[:, 1] == cluster][:,
                                                  0])  # Select all nodes of a cluster then choose one randomly
            controller_indices.append(cluster_controller)
        return controller_indices

    def _get_adjacent_clusters(self):
        """
        Gets which clusters are adjacent by iterating through all edges
        Returns Numpy adjacency matrix of clusters
        """
        adjacency_matrix = np.zeros(shape=(len(self.clusters), len(self.clusters)), dtype=int)
        graph_nodes = dict(self.graph.nodes(data='cluster'))
        for edge in self.graph.edges():
            # edge is (u, v) where u and v are node IDs
            # node_1 = self.graph.nodes[edge[0]]['id']
            # node_2 = self.graph.nodes[edge[1]]['id']
            node_1 = edge[0]
            node_2 = edge[1]
            if graph_nodes[node_1] != graph_nodes[node_2]:
                adjacency_matrix[graph_nodes[node_1], graph_nodes[node_2]] = 1
                adjacency_matrix[graph_nodes[node_2], graph_nodes[node_1]] = 1
        return adjacency_matrix

    def stepLA(self, graph: nx.Graph, controllers: list) -> int:
        """
        Helper function used to calculate distance between chosen controllers
        :param controllers: The list of chosen controllers in which to calculate distance between
        :return: The total distance from every controller to the other controllers
        """
        distance = 0
        for current_controller in range(len(controllers)):
            for other_controller in range(current_controller, len(controllers)):
                distance += nx.dijkstra_path_length(graph, controllers[current_controller],
                                                    controllers[other_controller])
        return distance

    def calculateOptimal(self) -> (list, int):
        """
        Goes through all possible combinations of valid controllers and find best one.
        Returns:
            (List of best nodes, Best distance possible)
        """
        combinations = list(itertools.product(*self.clusters))
        min_dist = 1000000
        min_combination = None
        for combination in combinations:
            dist = self.step(combination)
            if (dist < min_dist):
                min_dist = dist
                min_combination = combination
        return (min_combination, min_dist)

    def findGraphCentroid(self):
        """
        Finds the centroid of the environment graph
        Returns:
            Centroid node
            Lowest weight (int)
        """
        lowest_weight = 100000000
        best_node = -1
        for cur_node in self.graph.nodes:
            cur_weight = 0
            for other_node in self.graph.nodes:
                if other_node == cur_node:
                    continue
                cur_weight += nx.shortest_path_length(self.graph, cur_node, other_node, weight='weight')
            # print("This is the length to all other nodes:", cur_weight, cur_node)
            if cur_weight < lowest_weight:
                lowest_weight = cur_weight
                best_node = cur_node
        return best_node, lowest_weight

    def calculateDistance(self, actions) -> int:
        """Returns total distance of all controller edges in the current graph"""
        totalDist = 0
        for action in list(itertools.combinations(actions, 2)):
            # print("distance to find is: ", action)
            # print(nx.shortest_path_length(self.graph, action[0], action[1], weight='weight'))
            totalDist += nx.shortest_path_length(self.graph, action[0], action[1], weight='weight')
        return totalDist

    def graphCentroidAction(self) -> list:
        actions = []  # Set of Controllers
        centroid = nx.barycenter(self.graph, weight="weight")[0]  # Graph Centroid
        # Search each cluster for a controller to add to the actions variable
        for index, cluster in enumerate(self.clusters):
            # if the centroid is in the cluster being iterated over, skip that cluster
            if self.graph.nodes[centroid]['cluster'] == index:
                continue

            bestNode = None
            lowestDistance = 100000000
            # search for node in cluster with smallest distance to centroid node
            for node in cluster:
                if nx.shortest_path_length(self.graph, centroid, node, weight='weight') < lowestDistance:
                    lowestDistance = nx.shortest_path_length(self.graph, centroid, node, weight='weight')
                    bestNode = node
            actions.append(bestNode)

        bestNode = None
        lowestDistance = 10000000
        # search for best node in the cluster where the centroid node lies within
        for node in self.clusters[self.graph.nodes[centroid]['cluster']]:
            controller_graph, bad_controllers = self._set_controllers(actions + [node])
            distance = controller_graph.size(
                weight='weight') + 100000 * bad_controllers  # 100000 distance per invalid controller
            if distance < lowestDistance:
                lowestDistance = distance
                bestNode = node
        actions.append(bestNode)

        # Simulated Annealing Meta Heuristic
        current_state = actions
        annealing_rate = 0.90
        for x in range(1000):
            temperature = 1
            # choose a random already-selected node
            randomCluster = np.random.randint(len(self.clusters))
            randomBestAction = current_state[randomCluster]
            # get the neighbors of the randomly selected controller
            neighborList = [v for k, v in self.graph.edges if
                            (k == randomBestAction and v in self.clusters[
                                self.graph.nodes[randomBestAction]['cluster']]) or
                            (v == randomBestAction and k in self.clusters[
                                self.graph.nodes[randomBestAction]['cluster']])]
            neighborList.append(randomBestAction)
            # Get a random neighbor
            randomNeighbor = np.random.choice(neighborList)
            # create an alternate controller list with the neighbor replacing the previous controller
            proposed_state = current_state.copy()
            proposed_state[randomCluster] = randomNeighbor
            # get a random float between 0 and 1 to be the probability of changing the controller list
            threshold = float(np.random.rand(1))

            controller_graph, bad_controllers = self._set_controllers(proposed_state)
            proposed_distance = controller_graph.size(
                weight='weight') + 100000 * bad_controllers  # 100000 distance per invalid controller

            controller_graph, bad_controllers = self._set_controllers(current_state)
            current_distance = controller_graph.size(
                weight='weight') + 100000 * bad_controllers  # 100000 distance per invalid controller

            # if the new controller list is better than the original controller list, change to the new controller list
            # else change to the worst controller list with probability e^(-(new_distance - old_distance) / temperature)
            probability = 1 if proposed_distance < current_distance else np.exp(
                -(proposed_distance - current_distance) / temperature)
            if probability >= threshold:
                current_state = proposed_state.copy()
            temperature *= annealing_rate

        # ensure that the function returns the best of the original heuristic or the SA version
        controller_graph, bad_controllers = self._set_controllers(actions)
        original_distance = controller_graph.size(
            weight='weight') + 100000 * bad_controllers  # 100000 distance per invalid controller

        controller_graph, bad_controllers = self._set_controllers(current_state)
        SA_distance = controller_graph.size(
            weight='weight') + 100000 * bad_controllers  # 100000 distance per invalid controller

        if original_distance < SA_distance:
            print("Using original heuristic")
            return actions, original_distance
        else:
            print("Using SA heuristic")
            return current_state, SA_distance

    def compute_greedy_heuristic(self):
        """
        Computes WMSCP Greedy Heuristic. self.graph is the NetworkX graph. self.clusters is a list of lists of node incides
        where each list corresponds to the list of nodes in each cluster (so [[0,1,2,3],[4,5,6,7]] would be two clusters with 0-3
        in one cluster and 4-7 in other).
        """
        controller_set = []  # Stores controller indices
        distances = nx.floyd_warshall(self.graph, 'weight')  # Stores distances between every node
        weights_index = 0  # Converts index of minimum weight to actual node index (only works because clusters are in-order and sorted
        for i in range(len(self.clusters)):
            # Go through each node in cluster and compute weights
            weights = []
            for node in self.clusters[i]:
                # Go through every cluster and find shortest distance, shortest distance to controller
                weight = 0  # Stores weight of this node
                for j in range(len(self.clusters)):
                    if i == j:  # Matching cluster, ignore
                        continue
                    cluster_distance = 1e6
                    controller_distance = 0
                    for cluster_node in self.clusters[j]:
                        # Go through every node in other clusters and compute distance to find shortest node per cluster
                        cluster_distance = min(cluster_distance, distances[node][cluster_node])
                        if cluster_node in controller_set:
                            # If node is a controller, add its distance a "second" time to correspond
                            # to second sigma-sum of weight calculation
                            assert controller_distance == 0, print(cluster_node, controller_set)
                            controller_distance = distances[node][cluster_node]
                    assert cluster_distance != 1e6, "Cluster distance was too low, it was not modified"
                    weight += cluster_distance + controller_distance  # Add to node weight
                    weight /= len(self.clusters[
                                      i]) + 1  # Divide by nodes covered (the length of cluster upcoming controller is in)
                weights.append(weight)
            controller_set.append(weights.index(min(weights)) + weights_index)  # Add node index to controller list
            weights_index += len(self.clusters[i])
        return controller_set, self.step(controller_set)  # Return controller list and corresponding cost (minimum distance between all controllers)


def generateGraph(num_clusters: int, num_nodes: int, prob_cluster: float = 0.5, prob: float = 0.2, weight_low: int = 0,
                  weight_high: int = 100, draw=True) -> (nx.Graph, list, dict):
    """
    Generates graph given number of clusters and nodes
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

    # Test having first cluster have a huge weight
    first = True

    # Create clusters and add random edges within each cluster before merging them into single graph
    for i in range(num_clusters):
        # Add tree to serve as base of cluster subgraph. Loop through all edges and assign weights to each
        # cluster = nx.random_tree(nodes_per_cluster)
        p = 0.1  # TODO: Move to constants
        cluster = nx.fast_gnp_random_graph(nodes_per_cluster, p)
        while (not nx.is_connected(cluster)):
            cluster = nx.fast_gnp_random_graph(nodes_per_cluster, p)
        for start, end in cluster.edges:
            if first:
                cluster.add_edge(start, end, weight=1000)
            else:
                cluster.add_edge(start, end, weight=random.randint(weight_low, weight_high))
        first = False

        # Add edges to increase connectivity of cluster
        new_edges = np.random.randint(0, nodes_per_cluster, (int(nodes_per_cluster * prob_cluster), 2))
        new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
        new_edges = np.append(new_edges, new_weights, 1)
        cluster.add_weighted_edges_from(new_edges)

        # Set attributes and colors
        nx.set_node_attributes(cluster, i, 'cluster')
        node_colors[node_num:(node_num + nodes_per_cluster)] = i
        node_num += nodes_per_cluster
        clusters[i, :] = np.asarray(cluster.nodes) + nodes_per_cluster * i

        # Merge cluster with main graph
        G = nx.disjoint_union(G, cluster)

    # Add random edges to any nodes to increase diversity
    new_edges = np.random.randint(0, num_nodes, (int(num_nodes * 0.5), 2))
    new_weights = np.random.randint(weight_low, weight_high, (new_edges.shape[0], 1))
    new_edges = np.append(new_edges, new_weights, 1)
    G.add_weighted_edges_from(new_edges)

    # Add an edge to connect all clusters (to gurantee it is connected)
    node_num = nodes_per_cluster - 1 + nodes_per_cluster
    edge_weight = 1000
    G.add_edge(nodes_per_cluster - 1, nodes_per_cluster, weight=10000)
    for i in range(num_clusters - 1 - 1):
        G.add_edge(node_num, node_num + 1, weight=5)  # random.randint(weight_low, weight_high))
        node_num += nodes_per_cluster
        edge_weight = int(1000 * pow(0.2, i))

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
        cluster.edges[u, v]['weight'] = np.random.random() * 0.75 * (weight_high - weight_low) + weight_low + 0.25 * (
                    weight_high - weight_low)

    inner_cluster_edges = np.random.randint(0, len(clusters[0]),
                                            (int(np.log2(len(clusters[0]))), 2))

    # add edge weights to new edges of first cluster
    inner_cluster_edges = [
        (u, v, np.random.random() * 0.75 * (weight_high - weight_low) + weight_low + 0.25 * (weight_high - weight_low))
        for u, v in inner_cluster_edges]
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

    # Draw graph
    if False:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_labels(G, pos)
        # nx.draw_networkx_edge_labels(G, pos)
        nx.draw_networkx_edges(G, pos, G.edges())
        plt.draw()
        plt.show()

    return G, clusters, pos


def generateClusters(graph: nx.Graph, edge_label: str = None) -> (nx.Graph, list, dict):
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
    new_graph = nx.relabel.convert_node_labels_to_integers(
        graph)  # Converts node IDs to ints in case they weren't before
    clusters = list(nx.algorithms.community.greedy_modularity_communities(new_graph))
    # Add cluster attribute
    node_attrs = {}
    for i in range(len(clusters)):
        node_list = clusters[i]
        for node in node_list:
            node_attrs[node] = {'cluster': i}
    nx.set_node_attributes(new_graph, node_attrs)

    # Set random edge weights if no edge label is set
    if edge_label is None:
        for (u, v) in new_graph.edges():
            new_graph.edges[u, v]['weight'] = random.randint(0, 10)
    else:
        # Use LinkSpeed (unit GB/s) edge attribute as weight
        edge_dict = nx.get_edge_attributes(new_graph, edge_label)
        new_edges = {key: float(value) for key, value in edge_dict.items()}
        nx.set_edge_attributes(new_graph, new_edges, 'weight')

    return new_graph, clusters, nx.kamada_kawai_layout(new_graph)