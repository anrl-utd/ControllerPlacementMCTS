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

    CREATEGRAPHS = False # else it will read in graphs
    LOCATION = "Graphs/time_graph_files/3_08_2022_C=10/graph_24/"
    SEED = 4001
    NUMBER_ITERATIONS = 100
    PRINT = True


    start_time = time.time()
    finishtime1 = 0

    graph = None
    clusters = None
    pos = None

    if not os.path.isdir(LOCATION):
        os.mkdir(LOCATION)

    if CREATEGRAPHS or not (os.path.isfile(LOCATION+'clusters.pickle') and os.path.isfile(LOCATION+'graph.gpickle') and os.path.isfile(LOCATION+'position.pickle')):
        NUMBEROFNODES = 500
        NUMBEROFCLUSTERS = 10
        graph, clusters, pos = generateAlternateGraph(NUMBEROFCLUSTERS,NUMBEROFNODES)

        nx.write_gpickle(graph, LOCATION+'graph.gpickle')
        pickle.dump(clusters, open(LOCATION+'clusters.pickle', 'wb'))
        pickle.dump(pos, open(LOCATION+'position.pickle', 'wb'))
    else:
        graph = nx.read_gpickle(LOCATION+"graph.gpickle")
        clusters = pickle.load(open(LOCATION+"clusters.pickle", 'rb'))
        pos = pickle.load(open(LOCATION+"position.pickle", 'rb'))



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


    # Tracking best iteration
    max_score = -1000000
    max_controllers = []

    min_score = 1000000
    min_controllers = []

    score_controllers = []

    mcts_start_time = time.time()
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

    print("--- %s MCTS Runtime seconds ---" % (time.time() - mcts_start_time))
    sum = 0
    for iter in range(NUMBER_ITERATIONS):
        # print(score_controllers[i])
        sum = sum + score_controllers[iter][0]

    heuristic_time = time.time()
    heuristic_env = ControllerEnv(graph, clusters)

    heuristic_controllers, heuristic = heuristic_env.graphCentroidAction()
    print("--- %s Heuristic Runtime seconds ---" % (time.time() - heuristic_time))


    start = time.time()
    greedy_heuristic_controllers, greedy_heuristic = heuristic_env.compute_greedy_heuristic()
    print("--- %s Greedy Heuristic Runtime seconds ---" % (time.time() - start))




    # Print total results
    if PRINT:
        print("Average: " + str(-1*sum / NUMBER_ITERATIONS))
        print("Best Score: " + str(-1*max_score) + "---" + str(max_controllers))
        print("Worst Score: " + str(-1*min_score) + "---" + str(min_controllers))
        print("Heuristic: "+str(heuristic_controllers)+"  "+ str(heuristic))
        print("Greedy_Heuristic: " + str(greedy_heuristic_controllers) + "  " + str(greedy_heuristic))


        # opt_time = time.time()
        # print("optimal" + str(x.calculateOptimal()))
        # print("--- %s Optimal  Runtime seconds ---" % (time.time() - opt_time))
        heuristic_env.render()

    print("Finished test")



