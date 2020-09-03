import numpy as np
import itertools
import copy
import networkx as nx
from Examples import ControllerPlacement_env as game

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
            self.current_controllers[i]=-1
        #print("Initialized environment!")






# return a list of nodes that could be chosen,

# checks which clusters have selected a controller.
# adds all the nodes from clusters that lack a controller
# returns list
def GetActions(CurrentState):
    possibleActions = []
    i = 0
    for controller in np.nditer(CurrentState.current_controllers):
        if controller == -1:
            cluster = CurrentState.clusters[i]
            for node in cluster:
                possibleActions.append(node)
        i += 1

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

def ApplyAction(CurrentState, Action):
    state2 = game.State(CurrentState.clusters)
    state2.current_controllers = CurrentState.current_controllers.copy()
    state2.selectedControllers = CurrentState.selectedControllers

    clustersCopy = CurrentState.clusters.copy()  # made need to do something like np.array(list(CurrentState.clusters),dtype=np.int32)
    cluster = []
    for set in clustersCopy:
        cluster.append(list(set))
    clusterIndex = -1
    nodeIndex = -1

    for x_index, x in enumerate(cluster):
        for y_index, y in enumerate(x):
            if y == Action:
                nodeIndex = y_index
                clusterIndex = x_index
                break
        if nodeIndex != -1:
            break



    controller_graph = None  # Stores controller metagraph
    state2.selectedControllers += 1
    state2.current_controllers[clusterIndex] = Action
    return state2


    # -----------------------------------------------------------------------#
    # Description:
    #	Applies a random action to the current state and returns the next state
    # -----------------------------------------------------------------------#

def GetNextState(CurrentState):
    Actions = GetActions(CurrentState)
    i = np.random.randint(0, len(Actions))
    Action = Actions[i]
    NextState = ApplyAction(CurrentState, Action)
    return NextState


def IsTerminal(State):
    return State.selectedControllers == State.numberClusters


def GetStateRepresentation(State):
    return State.current_controllers


# Todo return state in string form

    # -----------------------------------------------------------------------#
    # Description:
    #	Returns list of possible nextStates from CurrentState
    # -----------------------------------------------------------------------#

def EvalNextStates(CurrentState):
    A = GetActions(CurrentState)
    NextStates = []
    # for i in range(len(A[:,0])):
    for i in range(len(A)):
        Action = A[i]
        NextState = ApplyAction(CurrentState, Action)
        NextStates.append(NextState)

    # if (A == []):
    #     NextState = ApplyAction(CurrentState, A)
    #     NextStates.append(NextState)

    return NextStates



def GetResult(CurrentState, graph: nx.graph):
    controller_graph = set_controllers(CurrentState,graph)
    # Return output reward

    return controller_graph.size(weight='weight')*-1


def set_controllers(CurrentState, graph: nx.graph):
    """
	Creates metagraph of controllers
	Args:
		controllers: Array of controller indices

	Returns:
		Complete graph of controllers (metagraph)
	"""

    found_clusters = np.zeros((len(CurrentState.clusters)))  # Stores what clusters have controllers been found for
    clusters = nx.get_node_attributes(graph, 'cluster')
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
        if _get_adjacent_clusters(CurrentState, graph)[first_cluster][second_cluster] == 1:
            controller_graph.add_edge(first_cluster, second_cluster,
                                      weight=nx.dijkstra_path_length(graph, source=pair[0], target=pair[1]))

    return controller_graph


def _get_adjacent_clusters(CurrentState, graph):
    """
    Gets which clusters are adjacent by iterating through all edges
    Returns Numpy adjacency matrix of clusters
    """
    adjacency_matrix = np.zeros(shape=(len(CurrentState.clusters), len(CurrentState.clusters)), dtype=int)
    graph_nodes = dict(graph.nodes(data='cluster'))
    for edge in graph.edges():
        # edge is (u, v) where u and v are node IDs
        # node_1 = self.graph.nodes[edge[0]]['id']
        # node_2 = self.graph.nodes[edge[1]]['id']
        node_1 = edge[0]
        node_2 = edge[1]
        if graph_nodes[node_1] != graph_nodes[node_2]:
            adjacency_matrix[graph_nodes[node_1], graph_nodes[node_2]] = 1
            adjacency_matrix[graph_nodes[node_2], graph_nodes[node_1]] = 1
    return adjacency_matrix

"""GetResult--Returns total distance between metagraph returned from setcontrollers
            (calculates distance between controllers)
"""





