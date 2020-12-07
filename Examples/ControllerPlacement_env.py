import numpy as np
import itertools
import copy
from Examples import ControllerPlacement_env as game
import networkx as nx
import time

# States are given as:
# bins = np.array([v1, v2,..., vn])
# state = np.array([v1, v2, v3, ..., vk])
MAX_VOLUME = 10.0


class State:
    def __init__(self, clusters: list, ):
        self.selectedControllers = 0
        self.clusters = clusters
        self.numberClusters = len(clusters)
        self.current_controllers = np.zeros((self.numberClusters,),
                                            dtype=int)  # Stores controllers placed in last action (used for rendering)
        for i in range(len(self.current_controllers)):
            self.current_controllers[i] = -1
        # print("Initialized environment!")


class ControllerPlacement_env:
    def __init__(self, Node, graph):
        self.controller_distances = {}
        self.graph = graph.copy()
        self.root = Node
        self.adjacencyMatrix = self._get_adjacent_clusters(Node.state)


    # checks which clusters have selected a controller.
    # adds all the nodes from clusters that lack a controller
    # returns list
    def GetActions(self, CurrentState):
        possibleActions = []

        cluster = CurrentState.clusters[CurrentState.selectedControllers]
        for node in cluster:
            possibleActions.append(node)

        # for each index i in CurrentState.controllers that  == 0
        # cluster CurrentState.clusters[i]
        # for each node in cluster
        # append to possibleActions
        return possibleActions

        # -----------------------------------------------------------------------#
        # Description:
        #	Returns a copy of currentState with the Action applied
        #   Action is node # to be a controller
        #   Sets the value at the index corresponding to the nodes cluster to the nodes value
        # -----------------------------------------------------------------------#

    def ApplyAction(self, CurrentState, Action):
        state2 = game.State(CurrentState.clusters)
        state2.current_controllers = CurrentState.current_controllers.copy()
        state2.selectedControllers = CurrentState.selectedControllers

        state2.selectedControllers += 1
        state2.current_controllers[CurrentState.selectedControllers] = Action
        return state2

        # -----------------------------------------------------------------------#
        # Description:
        #	Applies a random action to the current state and returns the next state
        # -----------------------------------------------------------------------#

    def GetNextState(self, CurrentState):
        Actions = self.GetActions(CurrentState)
        i = np.random.randint(0, len(Actions))
        Action = Actions[i]
        NextState = self.ApplyAction(CurrentState, Action)
        return NextState

    def IsTerminal(self, State):
        return State.selectedControllers == State.numberClusters

    def GetStateRepresentation(self, State):
        return State.current_controllers

    # Todo return state in string form

    # -----------------------------------------------------------------------#
    # Description:
    #	Returns list of possible nextStates from CurrentState
    # -----------------------------------------------------------------------#

    def EvalNextStates(self, CurrentState):
        A = self.GetActions(CurrentState)
        NextStates = []
        # for i in range(len(A[:,0])):
        for i in range(len(A)):
            Action = A[i]
            NextState = self.ApplyAction(CurrentState, Action)
            NextStates.append(NextState)

        # if (A == []):
        #     NextState = ApplyAction(CurrentState, A)
        #     NextStates.append(NextState)

        return NextStates

    def GetResult(self, CurrentState):
        controller_graph = self.set_controllers(CurrentState)
        # Return output reward
        Result = controller_graph.size(weight='weight') * -1
        print(CurrentState.current_controllers)
        print(Result)

        return Result

    def set_controllers(self, CurrentState):
        """
        Creates metagraph of controllers
        Args:
            controllers: Array of controller indices

        Returns:
            Complete graph of controllers (metagraph)
        """

        times = time.time()
        found_clusters = np.zeros((len(CurrentState.clusters)))  # Stores what clusters have controllers been found for
        clusters = nx.get_node_attributes(self.graph, 'cluster')
        index = 0


        valid_controllers = []
        for controller in CurrentState.current_controllers:
            # Multiple controllers in a cluster
            if found_clusters[clusters[controller]] == 0:
                found_clusters[clusters[controller]] = 1
                valid_controllers.append(controller)



        # Controllers were found to be valid. Now add controllers to complete metagraph.
        # new_contr_indices = []
        # for i in range(len(valid_controllers)):
        #     new_contr_indices.append([i, valid_controllers[i]])
        # controller_graph = nx.complete_graph(len(new_contr_indices))  # Store controller metagraph
        #
        # # Add edges between controllers in metagraph
        # for pair in itertools.combinations(new_contr_indices, 2):
        #     controller_graph.add_edge(pair[0][0], pair[1][0],
        #                               weight=nx.dijkstra_path_length(graph, source=pair[0][1],
        #                                                              target=pair[1][1]))

        controller_graph = nx.Graph()  # Store controller metagraph
        controller_graph.add_nodes_from(range(len(valid_controllers)))

        # Add edges between controllers in metagraph
        for pair in itertools.combinations(valid_controllers, 2):
            first_cluster = clusters[pair[0]]
            second_cluster = clusters[pair[1]]
            assert first_cluster != second_cluster, "2 controllers in same cluster in _set_controllers {} {}".format(
                first_cluster, second_cluster)
            if self.adjacencyMatrix[first_cluster][second_cluster] == 1:
                controller_graph.add_edge(first_cluster, second_cluster, weight=self._get_distance(pair[0], pair[1]))

        return controller_graph

    def _get_adjacent_clusters(self, CurrentState):
        """
        Gets which clusters are adjacent by iterating through all edges
        Returns Numpy adjacency matrix of clusters
        """
        adjacency_matrix = np.zeros(shape=(len(CurrentState.clusters), len(CurrentState.clusters)), dtype=int)
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

    def _get_distance(self, controller_1, controller_2):
        """
            Returns distance between two controllers and uses dynamic programming to save some computation time
            """
        less_controller = None
        greater_controller = None
        if controller_1 < controller_2:
            less_controller = controller_1
            greater_controller = controller_2
        else:
            less_controller = controller_2
            greater_controller = controller_1
        if (less_controller, greater_controller) in self.controller_distances:
            return self.controller_distances[(less_controller, greater_controller)]
        else:
            distance = nx.dijkstra_path_length(self.graph, source=less_controller, target=greater_controller)
            self.controller_distances[(less_controller, greater_controller)] = distance
            return distance



