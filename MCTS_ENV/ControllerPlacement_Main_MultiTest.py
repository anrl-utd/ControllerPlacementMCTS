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


    CREATEGRAPHS = True ; # else it will read in graphs
    GRAPH_DIR_LOCATION = "Graphs/time_graph_files/3_08_2022_C=5"
    GRAPHS_LOCATION = GRAPH_DIR_LOCATION + "/graph_{}"
    RESULTS_LOCATION = "Graphs/time_graph_results/3_08_2022_C=5_results.{}.csv"
    NUMBER_OF_GRAPHS = 40

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

    for i in range(0, NUMBER_OF_GRAPHS+0):

        NUMBEROFNODES = 100 + 25*(i+1)
        NUMBEROFCLUSTERS = 5

        j = i

        if CREATEGRAPHS:
            graph, clusters, pos = generateAlternateGraph(NUMBEROFCLUSTERS,NUMBEROFNODES)
            if not os.path.isdir(GRAPH_DIR_LOCATION):
                os.mkdir(GRAPH_DIR_LOCATION)

            if not os.path.isdir(GRAPHS_LOCATION.format(j)):
                os.mkdir(GRAPHS_LOCATION.format(j))

            nx.write_gpickle(graph, GRAPHS_LOCATION.format(j)+'/graph.gpickle')
            pickle.dump(clusters, open(GRAPHS_LOCATION.format(j)+'/clusters.pickle', 'wb'))
            pickle.dump(pos, open(GRAPHS_LOCATION.format(j)+'/position.pickle', 'wb'))
        else:
            graph = nx.read_gpickle(GRAPHS_LOCATION.format(j)+"/graph.gpickle")
            clusters = pickle.load(open(GRAPHS_LOCATION.format(j)+"/clusters.pickle", 'rb'))
            pos = pickle.load(open(GRAPHS_LOCATION.format(j)+"/position.pickle", 'rb'))


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

        MCTS_time = time.time()
        for k in range(NUMBER_ITERATIONS):
            if PRINT:
                print("iteration: " + str(k)+"\n")

            iter_time = time.time()
            # Set to true to see every iteation of a single MCTS test
            prints = False  # print each iteration

            # Generating Environment for test
            RootState = game.State(clusters)
            Root = nd.Node(RootState)
            environment = game.ControllerPlacement_env(Root, graph, prints)

            # Running MCTS Test with generated environment
            x = MCTS.MCTS(environment, False, prints)
            x.Run()

            # Tracking results of MCTS test run
            MCTS_iter_time = time.time() - iter_time
            score_controllers.append(
                [x.maxScore, x.maxControllers, "--- %s Runtime seconds ---" % (MCTS_iter_time)])
            if PRINT:
                print([x.maxScore, x.maxControllers, "--- %s Runtime seconds ---" % (MCTS_iter_time)])

            # Tracking best score
            if x.maxScore > max_score:
                max_score = x.maxScore
                max_controllers = x.maxControllers

            if x.maxScore < min_score:
                min_score = x.maxScore
                min_controllers = x.maxControllers
            if True:
                print("--- %s Runtime seconds ---" % (time.time() - iter_time))

        sum = 0
        for iter in range(NUMBER_ITERATIONS):
            # print(score_controllers[i])
            sum = sum + score_controllers[iter][0]

        MCTS_time =  time.time() - MCTS_time

        heuristic_time = time.time()
        heuristic_env = ControllerEnv(graph, clusters)
        heuristic_controllers, heuristic = heuristic_env.graphCentroidAction()
        heuristic_time = time.time() - heuristic_time
        print("--- %s Heuristic Runtime seconds ---" % (heuristic_time))

        greedy_heuristic_time = time.time()
        greedy_heuristic_controllers, greedy_heuristic = heuristic_env.compute_greedy_heuristic()
        greedy_heuristic_time = time.time() - greedy_heuristic_time
        print("--- %s Greedy Heuristic Runtime seconds ---" % (greedy_heuristic_time))

        # Print total results
        if PRINT:
            print("Average: " + str(-1*sum / NUMBER_ITERATIONS))
            print("Best Score: " + str(-1*max_score) + "---" + str(max_controllers))
            print("Worst Score: " + str(-1*min_score) + "---" + str(min_controllers))
            print("Heuristic: "+str(heuristic_controllers)+"  "+ str(heuristic))
            print("Greedy_Heuristic: " + str(greedy_heuristic_controllers) + "  " + str(greedy_heuristic))


        writer.writerow([i,str(-1*sum / NUMBER_ITERATIONS), str(-1*max_score) , str(-1*min_score), str(max_controllers), str(min_controllers),str(heuristic), str(heuristic_controllers),str(greedy_heuristic), str(greedy_heuristic_controllers),str(MCTS_time),str(heuristic_time),str(greedy_heuristic_time)])

        print("Finished test: {}".format(i))
    file.close()
    print("FINISHED")

