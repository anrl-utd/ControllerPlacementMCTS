import numpy as np
import itertools
import copy
import networkx as nx

# States are given as:
# bins = np.array([v1, v2,..., vn])
# state = np.array([v1, v2, v3, ..., vk])
MAX_VOLUME = 10.0


class State:
    def __init__(self, graph: nx.Graph, clusters: list, pos: dict = None):
        self.selectedControllers = 0
        self.original_graph = graph.copy()  # Keep original graph in case of needing it for reset
        # Generate graph display positions if needed
        if (pos is None):
            self.pos = nx.kamada_kawai_layout(graph)  # get the positions of the nodes of the graph
        else:
            self.pos = pos
        self.clusters = clusters
        self.graph = graph.copy()
        self.numberClusters = len(clusters)
        self.current_controllers = np.zeros((self.numberClusters,),
                                            dtype=int)  # Stores controllers placed in last action (used for rendering)
        for i in range(len(self.current_controllers)):
            self.current_controllers[i]=-1
        print("Initialized environment!")

    def _graph_degree(self):
        """Returns the highest degree of a node in the graph"""
        return max([degree for node, degree in self.graph.degree()])




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
    state2 = copy.deepcopy(CurrentState)  # made need to make a copy function for state.
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
    return 1


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

    if (A == []):
        NextState = ApplyAction(CurrentState, A)
        NextStates.append(NextState)

    return NextStates


def set_controllers(CurrentState):
    """
	Creates metagraph of controllers
	Args:
		controllers: Array of controller indices

	Returns:
		Complete graph of controllers (metagraph)
	"""

    found_clusters = np.zeros((len(CurrentState.clusters)))  # Stores what clusters have controllers been found for
    clusters = nx.get_node_attributes(CurrentState.graph, 'cluster')
    index = 0

    valid_controllers = []
    for controller in CurrentState.current_controllers:
        # Multiple controllers in a cluster
        if found_clusters[clusters[controller]] == 0:
            found_clusters[clusters[controller]] = 1
            valid_controllers.append(controller)

    # Controllers were found to be valid. Now add controllers to complete metagraph.
    new_contr_indices = []
    for i in range(len(valid_controllers)):
        new_contr_indices.append([i, valid_controllers[i]])
    controller_graph = nx.complete_graph(len(new_contr_indices))  # Store controller metagraph

    # Add edges between controllers in metagraph
    for pair in itertools.combinations(new_contr_indices, 2):
        controller_graph.add_edge(pair[0][0], pair[1][0],
                                  weight=nx.dijkstra_path_length(CurrentState.graph, source=pair[0][1],
                                                                 target=pair[1][1]))
    return controller_graph


"""GetResult--Returns total distance between metagraph returned from setcontrollers
            (calculates distance between controllers)
"""


def GetResult(CurrentState):
    controller_graph = set_controllers(CurrentState)
    # Return output reward

    return controller_graph.size(weight='weight')*-1




def step(self, action: list) -> int:

    controller_graph = None  # Stores controller metagraph
    # Create metagraph of controllers. The node at an index corresponds to the controller of the cluster of that index
    controller_graph = self.set_controllers(action)
    # Return output reward
    return controller_graph.size(weight='weight')

def calculateOptimal(self, state) -> (list, int):
    """
    Goes through all possible combinations of valid controllers and find best one.
    Returns:
        (List of best nodes, Best distance possible)
    """
    combinations = list(itertools.product(*state.clusters))
    min_dist = 1000000
    min_combination = None
    for combination in combinations:
        dist = self.step(combination)
        if (dist < min_dist):
            min_dist = dist
            min_combination = combination
    return (min_combination, min_dist)