import os
import pickle5 as pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import MCTS_ENV.ControllerPlacement_MCTS_env as game
import Node as nd
import numpy as np
import MCTS
import random
import time
import csv
import os
from MCTS_ENV.ControllerEnv import generateGraph, generateAlternateGraph, generateClusters, ControllerEnv


if __name__ == "__main__":

    CREATEGRAPHS = True; # else it will read in graphs
    GRAPHS_LOCATION = "Graphs/graph_files/graph_{}"
    RESULTS_LOCATION = "Graphs/graph_results/graph_results_{}.csv"
    NUMBER_OF_GRAPHS = 20

    start_time = time.time()
    finishtime1 = 0

    graph = None
    clusters = None
    pos = None

    num = 0
    while os.path.isfile(RESULTS_LOCATION.format(num)):
        num+=1

    file = open(RESULTS_LOCATION.format(num), 'w', newline='')
    writer = csv.writer(file)

    for i in range(1, NUMBER_OF_GRAPHS + 1):

        NUMBEROFNODES = i * 100
        NUMBEROFCLUSTERS = 5


        if CREATEGRAPHS:
            graph, clusters, pos = generateAlternateGraph(NUMBEROFCLUSTERS,NUMBEROFNODES)
            if not os.path.isdir(GRAPHS_LOCATION.format(i)):
                os.mkdir(GRAPHS_LOCATION.format(i))

            nx.write_gpickle(graph, GRAPHS_LOCATION.format(i)+'/graph.gpickle')
            pickle.dump(clusters, open(GRAPHS_LOCATION.format(i)+'/clusters.pickle', 'wb'))
            pickle.dump(pos, open(GRAPHS_LOCATION.format(i)+'/position.pickle', 'wb'))
        else:
            graph = nx.read_gpickle(GRAPHS_LOCATION.format(i)+"/graph.gpickle")
            clusters = pickle.load(open(GRAPHS_LOCATION.format(i)+"/clusters.pickle", 'rb'))
            pos = pickle.load((GRAPHS_LOCATION.format(i)+"/position.pickle", 'rb'))


        print("\n\n\nGRAPH: {}".format(i) + "\n")


        # Creating Clusters Array from only graph file

        # cluster_info = nx.get_node_attributes(graph, 'cluster')
        # clusters_array = np.array(list(cluster_info.items()),
        #                           dtype=np.int32)  # Construct numpy array as [[node num, cluster num], [..]]
        #
        # number_of_clusters = clusters_array[len(clusters_array) - 1][1] + 1
        #
        # rows = number_of_clusters
        # clusters = [[-1] for _ in range(rows)]
        #
        # for index in range(len(clusters_array)):
        #     node = clusters_array[index][0]
        #     cluster = clusters_array[index][1]
        #
        #     if clusters[cluster][0] == -1:
        #         clusters[cluster][0] = node
        #     else:
        #         clusters[cluster].append(node)

        # Number of times that MCTS is run for a given graph on different seeds
        NUMBER_ITERATIONS = 15
        PRINT = True

        # Tracking best iteration
        max_score = -1000000
        max_controllers = []

        min_score = 1000000
        min_controllers = []

        score_controllers = []

        for k in range(NUMBER_ITERATIONS):
            if PRINT:
                print("iteration: " + str(k)+"\n")
            iter_time = time.time()
            np.random.seed(k + 65)

            # Set to true to see every iteation of a single MCTS test
            prints = False  # print each iteration

            # Generating Environment for test
            RootState = game.State(clusters)
            Root = nd.Node(RootState)
            start_time = time.time()
            environment = game.ControllerPlacement_env(Root, graph, prints)

            # Running MCTS Test with generated environment
            x = MCTS.MCTS(environment, False, prints)
            x.Run()

            # Tracking results of MCTS test run
            score_controllers.append(
                [x.maxScore, x.maxControllers, "--- %s Runtime seconds ---" % (time.time() - iter_time)])
            if PRINT:
                print([x.maxScore, x.maxControllers, "--- %s Runtime seconds ---" % (time.time() - iter_time)])

            # Tracking best score
            if x.maxScore > max_score:
                max_score = x.maxScore
                max_controllers = x.maxControllers

            if x.maxScore < min_score:
                min_score = x.maxScore
                min_controllers = x.maxControllers
            if PRINT:
                print("--- %s Runtime seconds ---" % (time.time() - iter_time))

        sum = 0
        for iter in range(NUMBER_ITERATIONS):
            # print(score_controllers[i])
            sum = sum + score_controllers[iter][0]

        heuristic_env = ControllerEnv(graph, clusters)
        heuristic_controllers, heuristic = heuristic_env.graphCentroidAction()
        greedy_heuristic_controllers, greedy_heuristic = heuristic_env.compute_greedy_heuristic()

        # Print total results
        if PRINT:
            print("Average: " + str(-1*sum / NUMBER_ITERATIONS))
            print("Best Score: " + str(-1*max_score) + "---" + str(max_controllers))
            print("Worst Score: " + str(-1*min_score) + "---" + str(min_controllers))
            print("Heuristic: "+str(heuristic_controllers)+"  "+ str(heuristic))
            print("Greedy_Heuristic: " + str(greedy_heuristic_controllers) + "  " + str(greedy_heuristic))

        # file.write("Average: " + str(sum / NUMBER_ITERATIONS)+"\n")
        # file.write("Best Score: " + str(max_score) + "---" + str(max_controllers)+"\n")
        # file.write("Worst Score: " + str(min_score) + "---" + str(min_controllers)+"\n")
        writer.writerow([i,str(-1*sum / NUMBER_ITERATIONS), str(-1*max_score) , str(-1*min_score), str(max_controllers), str(min_controllers),str(heuristic), str(heuristic_controllers),str(greedy_heuristic), str(greedy_heuristic_controllers)])

        print("Finished test: {}".format(i))
    file.close()
    print("FINISHED")

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
