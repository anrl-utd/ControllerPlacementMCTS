# ------------------------------------------------------------------------#
#
# Written by sergeim19 (Created June 21, 2017)
# https://github.com/sergeim19/
# Last Modified Aug 7, 2017
#
# Description:
# Single Player Monte Carlo Tree Search implementation.
# This is a Python implementation of the single player
# Monte Carlo tree search as described in the paper:
# https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf
#
# ------------------------------------------------------------------------#
import copy
import Node as nd
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import networkx as nx
import itertools
import time

# Import your game implementation here.
from MCTS_ENV import ControllerPlacement_MCTS_env as game


# ------------------------------------------------------------------------#
# Class for Single Player Monte Carlo Tree Search implementation.
# ------------------------------------------------------------------------#
class MCTS:

    # -----------------------------------------------------------------------#
    # Description: Constructor.
    # Node 	  - Root node of the tree of class Node.
    # Verbose - True: Print details of search during execution.
    # 			False: Otherwise
    # -----------------------------------------------------------------------#
    def __init__(self, env: game, Verbose=False,prints=False):
        self.verbose = Verbose

        # tracks best controller set and best score
        self.maxControllers = []
        self.maxScore = -100000
        self.environment = env
        self.prints = prints

    # -----------------------------------------------------------------------#
    # Description: Performs selection phase of the MCTS.
    # -----------------------------------------------------------------------#
    def Selection(self):
        SelectedChild = self.environment.root
        HasChild = False

        # Check if child nodes exist.
        if len(SelectedChild.children) > 0:
            HasChild = True
        else:
            HasChild = False

        while HasChild:
            SelectedChild = self.SelectChild(SelectedChild)
            if len(SelectedChild.children) == 0:
                HasChild = False
        # SelectedChild.visits += 1.0

        if self.verbose:
            print("\nSelected: ", self.environment.GetStateRepresentation(SelectedChild.state))

        return SelectedChild

    # -----------------------------------------------------------------------#
    # Description:
    # Given a Node, selects the first unvisited child Node, or if all
    # 	children are visited, selects the Node with greatest UTC value.
    # Node	- Node from which to select child Node from.
    # -----------------------------------------------------------------------#
    def SelectChild(self, Node):
        if len(Node.children) == 0:
            return Node

        for Child in Node.children:
            if Child.visits > 0.0:
                continue
            else:
                if self.verbose:
                    print("Considered child", self.environment.GetStateRepresentation(Child.state), "UTC: inf", )
                return Child

        MaxWeight = -1000000000000
        for Child in Node.children:
            # Weight = self.EvalUTC(Child)
            Weight = Child.sputc
            if self.verbose:
                print("Considered child:", self.environment.GetStateRepresentation(Child.state), "UTC:", Weight)
            if Weight > MaxWeight:
                MaxWeight = Weight
                SelectedChild = Child
        return SelectedChild

    # -----------------------------------------------------------------------#
    # Description: Performs expansion phase of the MCTS.
    # Leaf	- Leaf Node to expand.
    # -----------------------------------------------------------------------#
    def Expansion(self, Leaf):
        if self.IsTerminal(Leaf):
            if self.verbose:
                print("Is Terminal.")
            return False
        elif Leaf.visits == 0:  # has never been visited
            return Leaf
        else:
            # Expand.
            if len(Leaf.children) == 0:  # adds children to node
                Children = self.EvalChildren(Leaf)

                for NewChild in Children:
                    if np.all(NewChild.state == Leaf.state):  # comparator for state
                        continue
                    Leaf.AppendChild(NewChild)
            assert (len(Leaf.children) > 0), "Error"
            Child = self.SelectChildNode(Leaf)

        if self.verbose:
            print("Expanded: ", self.environment.GetStateRepresentation(Child.state))
        return Child

    # -----------------------------------------------------------------------#
    # Description: Checks if a Node is terminal (it has no more children).
    # Node	- Node to check.
    # -----------------------------------------------------------------------#
    def IsTerminal(self, Node):
        # Evaluate if node is terminal.
        if self.environment.IsTerminal(Node.state):
            return True
        else:
            return False

    # -----------------------------------------------------------------------#
    # Description:
    #	Evaluates all the possible children states given a Node state
    #	and returns the possible children Nodes.
    # Node	- Node from which to evaluate children.
    # -----------------------------------------------------------------------#
    def EvalChildren(self, Node):
        NextStates = self.environment.EvalNextStates(Node.state)
        Children = []
        for State in NextStates:
            ChildNode = nd.Node(State)
            Children.append(ChildNode)

        return Children

    # -----------------------------------------------------------------------#
    # Description:
    # Selects a child node randomly.
    # Node	- Node from which to select a random child.
    # -----------------------------------------------------------------------#
    def SelectChildNode(self, Node):
        # Randomly selects a child node.
        Len = len(Node.children)
        assert Len > 0, "Incorrect length"
        i = np.random.randint(0, Len)
        return Node.children[i]

    # -----------------------------------------------------------------------#
    # Description:
    #Performs the simulation phase of the MCTS.
    # Node	- Node from which to perform simulation.
    # -----------------------------------------------------------------------#
    def Simulation(self, Node):
        CurrentState = game.State(Node.state.clusters)

        CurrentState.current_controllers = Node.state.current_controllers.copy()

        CurrentState.selectedControllers = Node.state.selectedControllers
        # if(any(CurrentState) == False):
        #	return None
        if self.verbose:
            print("Begin Simulation")

        Level = self.GetLevel(Node)
        # Perform simulation.

        while not (self.environment.IsTerminal(CurrentState)):
            CurrentState = self.environment.GetNextState(CurrentState)
            Level += 1.0
            if self.verbose:
                print("CurrentState:", self.environment.GetStateRepresentation(CurrentState))
                # self.environment.PrintTablesScores(CurrentState)
        copytime = time.time()
        Result = self.environment.GetResult(CurrentState)
        # print("Sim time:" + str(time.time() - copytime))

        if Result > self.maxScore:
            self.maxScore = Result
            self.maxControllers = CurrentState.current_controllers

        # self.PrintResult(str(Result)+" Controllers: "+str(CurrentState.current_controllers))

        return Result

    # -----------------------------------------------------------------------#
    # Description:
    #	Performs the backpropagation phase of the MCTS.
    # Node		- Node from which to perform Backpropagation.
    # Result	- Result of the simulation performed at Node.
    # -----------------------------------------------------------------------#
    def Backpropagation(self, Node, Result):
        # Update Node's weight.

        CurrentNode = Node
        CurrentNode.wins += Result
        CurrentNode.ressq += Result ** 2
        CurrentNode.visits += 1


        while self.HasParent(CurrentNode):
            # Update parent node's weight.
            previousNode = CurrentNode
            CurrentNode = CurrentNode.parent

            CurrentNode.wins += Result
            CurrentNode.ressq += Result ** 2
            CurrentNode.visits += 1

            NodesToUpdate = CurrentNode.children
            for node in NodesToUpdate:
                if node.visits > 0:
                    self.EvalUTC(node)





    # self.root.wins += Result
    # self.root.ressq += Result**2
    # self.root.visits += 1
    # self.EvalUTC(self.root)

    # -----------------------------------------------------------------------#
    # Description:
    #	Checks if Node has a parent..
    # Node - Node to check.
    # -----------------------------------------------------------------------#
    def HasParent(self, Node):
        return Node.parent is not None


    # -----------------------------------------------------------------------#
    # Description:
    #	Evaluates the Single Player modified UTC. See:
    #	https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf
    # Node - Node to evaluate.
    # -----------------------------------------------------------------------#
    def EvalUTC(self, Node):
        # c = np.sqrt(2)
        c = 100
        w = Node.wins
        n = Node.visits
        sumsq = Node.ressq
        if Node.parent is None:
            t = Node.visits
        else:
            t = Node.parent.visits

        UTC = w / n + c * np.sqrt(np.log(t)/n)

        Node.sputc = UTC

        return Node.sputc
        # D = 0
        #
        # Modification = np.sqrt((sumsq - n * (w / n) ** 2 + D) / n)
        # # print "Original", UTC
        # # print "Mod", Modification
        # if np.isnan(Modification):
        #     Modification = 0


    # -----------------------------------------------------------------------#
    # Description:
    #	Gets the level of the node in the tree.
    # Node - Node to evaluate the level.
    # -----------------------------------------------------------------------#
    def GetLevel(self, Node):
        Level = 0.0
        while Node.parent:
            Level += 1.0
            Node = Node.parent
        return Level

    # -----------------------------------------------------------------------#
    # Description:
    #	Prints the tree to file.
    # -----------------------------------------------------------------------#
    def PrintTree(self):
        f = open('Tree.txt', 'w')
        Node = self.environment.root
        self.PrintNode(f, Node, "", False)
        f.close()

    # -----------------------------------------------------------------------#
    # Description:
    #	Prints the tree Node and its details to file.
    # Node			- Node to print.
    # Indent		- Indent character.
    # IsTerminal	- True: Node is terminal. False: Otherwise.
    # -----------------------------------------------------------------------#
    def PrintNode(self, file, Node, Indent, IsTerminal):
        file.write(Indent)
        if IsTerminal:
            file.write("\-")
            Indent += "  "
        else:
            file.write("|-")
            Indent += "| "

        string = str(self.GetLevel(Node)) + ") (["
        # for i in Node.state.bins: # game specific (scrap)
        # 	string += str(i) + ", "
        string += str(self.environment.GetStateRepresentation(Node.state))
        string += "], W: " + str(Node.wins) + ", N: " + str(Node.visits) + ", UTC: " + str(Node.sputc) + ") \n"
        file.write(string)

        for Child in Node.children:
            self.PrintNode(file, Child, Indent, self.IsTerminal(Child))




    # dont need actually. Instead in backpropagation just increment up and then update the UTC foreach sibling with array
    def checkUTCForEach(self, root: nd):
        arr = root.children

        for node in arr:
            if node.visits > 0: # and not node.isTerminal:
                self.EvalUTC(node)

                # if self.IsTerminal(node) or (len(node.children) > 0 and all( child.isTerminal is True for child in node.children)):
                #     node.isTerminal = True
                #     node.sputc = -10000000
                if len(node.children) > 0:
                    self.checkUTCForEach(node)





    def PrintResult(self, Result):
        filename = 'Results.txt'
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        f = open(filename, append_write)
        f.write(str(Result) + '\n')
        f.close()

    def calculateOptimal(self) -> (list, int):
        """
        Goes through all possible combinations of valid controllers and find best one.
        Returns:
            (List of best nodes, Best distance possible)
        """

        clustersCopy = self.environment.root.state.clusters.copy()  # made need to do something like np.array(list(CurrentState.clusters),dtype=np.int32)
        clusters = []
        for set in clustersCopy:
            clusters.append(list(set))

        combinations = list(itertools.product(*clusters))
        max_dist = -1000000
        min_combination = None
        print(len(combinations))
        for i, combination in enumerate(combinations):
            # print(i, "  ", max_dist)
            newState = game.State(self.environment.root.state.clusters)
            newState.current_controllers = combination

            dist = self.environment.GetResult(newState)
            if (dist > max_dist):
                max_dist = dist
                min_combination = combination
        return (min_combination, max_dist)

    # -----------------------------------------------------------------------#
    # Description:
    #	Runs the SP-MCTS.
    # MaxIter	- Maximum iterations to run the search algorithm.
    # -----------------------------------------------------------------------#
    def Run(self, MaxIter=20000,prints=False):
        self.prints = prints
        start_time0 = time.time()
        # nS = game.State(self.root.state.clusters)

        # arr = [ 62, 153, 254, 386, 495, 564, 656, 783, 880, 968]
        # nS.current_controllers = arr
        # print("TestScore")
        # print(self.environment.GetResult(nS, self.graph))

        # print("optimal"+str(self.calculateOptimal()))
        self.maxControllers = []

        y_list = []
        t_list = []
        minmax = -10000
        self.verbose = False
        for i in range(MaxIter):
            start_time = time.time()

            # if i != 0:
            #     self.checkUTCForEach(self.environment.root)
            if prints:
                print("\n===== Begin iteration:", i, "=====")
            X = self.Selection()

            Y = self.Expansion(X)

            if Y:

                Result = self.Simulation(Y)

                if self.verbose:
                    print("Result: ", Result)

                self.Backpropagation(Y, Result)

                y_list.append(Result)
            else:
                Result = self.environment.GetResult(X.state)
                y_list.append(Result)

                if self.verbose:
                    print(X.state.current_controllers)
                    print(Result)

                self.Backpropagation(X, Result)

                if Result > self.maxScore:
                    self.maxScore = Result
                    self.maxControllers = X.state.current_controllers

            t_list.append(time.time() - start_time)
            # print("--- %s seconds ---" % (time.time() - start_time))
            # self.PrintResult(Result)

        if prints:
            print("----Finished----")
            print("--- %s Total seconds ---" % (time.time() - start_time0))
            print("score:" + str(self.maxScore))
            print("max controllers: ")
            print(self.maxControllers)

            print("Search complete.")
            print("Iterations:", i)

            plt.plot([i for i in range(MaxIter)], y_list)
            plt.title(' Score Vs Iteration Step')
            plt.xlabel('Iteration Step')
            plt.ylabel('Max Score')
            plt.show()

            plt.plot([i for i in range(MaxIter)], t_list)
            plt.title('Time Vs Iteration Step')
            plt.xlabel('Iteration Step')
            plt.ylabel('Max Score')
            plt.show()
