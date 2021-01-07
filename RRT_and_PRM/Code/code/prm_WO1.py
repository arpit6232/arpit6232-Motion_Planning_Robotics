import matplotlib.pyplot as plt 
import numpy as np 
from scipy import spatial 
# https://www.geeksforgeeks.org/union-find/
from networkx.utils import UnionFind 
from shapely.geometry import Point, LineString, Polygon, \
    MultiPolygon
import random
import copy 
from graph import Graph
from workspaces import config
import yaml
import itertools
from timeit import default_timer as timer
import pandas as pd 
import seaborn as sns
import os

WORKSPACE_CONFIG = config()

class PRM:

    def __init__(self, name = None, n = None, r = None, smoothing = None):
        with open(name, 'r') as stream:
            configData = yaml.load(stream, Loader=yaml.Loader)

        # self.n = configData['n']
        # self.r = configData['r']
        # self.usePathSmoothing = [False, True]
        self.n = n
        self.r = r
        self.usePathSmoothing = smoothing
        self.minBounds = configData['minBounds']
        self.maxBounds = configData['maxBounds']
        self.iterations = 1000

    def check_for_obs_collission(self, startState, goalState):

        start = tuple(startState.flatten())
        goal = tuple(goalState.flatten())

        line = LineString([start, goal])

        obstacles = WORKSPACE_CONFIG['WO1']
        
        collide_flag = False
        for obstacle in obstacles:
            if obstacle.intersects(line):

                collide_flag = True

        return collide_flag


    def checkConnectivity(self, data_structure, currLabel, nbrLabel):

        currComponent = data_structure[currLabel]
        newComponent  = data_structure[nbrLabel]

        flag = (currComponent != newComponent)
        return flag 

    def admissible_heuristic_dist(self, point, dest, distNorm=2):
        p1 = np.reshape(point, (len(point), 1))
        p2 = np.reshape(dest, (len(dest), 1))

        distance = np.linalg.norm(p2 - p1, ord=distNorm)

        return distance

    def smoothPathInGraph(self, graph, path, goal_node_idx, pathLength,
                          shouldBenchmark):

        developed_path = copy.deepcopy(path)

        numEdgesToSmooth = round(len(developed_path) / 5)

        for i in range(0, numEdgesToSmooth):

            # only allow sampling from the middle of the path
            rNodes = tuple(self.replace_sample(developed_path[1:-1], 2))
            start_node_idx = rNodes[0]
            end_node_idx = rNodes[1]

            # skip the sampled nodes if they're already directly connected
            nodeBeforeEnd = graph.getter_helper(end_node_idx, 'prev')
            if nodeBeforeEnd == start_node_idx:
                continue

            # obtain the collision free samples
            startNodePos = graph.getter_helper(start_node_idx, 'pos')
            endNodePos = graph.getter_helper(end_node_idx, 'pos')

            collided = True
            itr = 0
            flag = False

            while collided and itr <= self.iterations:

                potentialSample = np.random.uniform(low=self.minBounds, high=self.maxBounds, size=(1, len(self.minBounds)))
                potentialSample = potentialSample.flatten()

                collided = self.check_for_obs_collission(startNodePos, potentialSample)\
                    or self.check_for_obs_collission(potentialSample, endNodePos)

                itr += 1

            if not collided:
                flag = True

            if not flag:
                continue

            # add the node to the PRM graph
            new_node_idx = goal_node_idx + i + 1
            goal_node_pos = graph.getter_helper(goal_node_idx, 'pos')
            updated_heuristic = self.admissible_heuristic_dist(potentialSample, goal_node_pos)
            graph.add_node(new_node_idx,
                           heuristic=updated_heuristic,
                           prev=start_node_idx,
                           dist=0, priority=0, pos=potentialSample)

            # connect it to the graph
            graph.add_edge(start_node_idx, new_node_idx, weight=self.admissible_heuristic_dist(startNodePos, potentialSample))

            graph.add_edge(new_node_idx, end_node_idx, weight=self.admissible_heuristic_dist(potentialSample, endNodePos))

            # remove in-between nodes on the path
            currNode = end_node_idx
            prev_node = graph.getter_helper(currNode, 'prev')

            while prev_node != start_node_idx:

                prevPrevNode = graph.getter_helper(prev_node, 'prev')
                developed_path.remove(prev_node)

                # need to update prev now in order to continue proper traversal
                graph.setter_helper(currNode, 'prev', prevPrevNode)

                # now set the linked list pointers
                prev_node = graph.getter_helper(prev_node, 'prev')

            # now insert the new node into its place
            endNodeIDX = developed_path.index(end_node_idx)
            developed_path.insert(endNodeIDX, new_node_idx)
            graph.setter_helper(end_node_idx, 'prev', new_node_idx)

        # compute new path length
        newPathEdges = graph.getPathEdges(developed_path)
        newPathLength = 0
        for edge in newPathEdges:
            newPathLength += graph.edges[edge]['weight']

        # only return the smoothed path if its shorter
        if newPathLength > pathLength:

            if not shouldBenchmark:
                print('smoothing failed, using unsmoothed path')

            return path, pathLength

        else:

            return developed_path, newPathLength

    def replace_sample(self, seq, sampleSize):

        totalElems = len(seq)

        picksRemaining = sampleSize
        for elemsSeen, element in enumerate(seq):
            elemsRemaining = totalElems - elemsSeen
            prob = picksRemaining / elemsRemaining
            if random.random() < prob:
                yield element
                picksRemaining -= 1

    def computePRM(self, startState, goalState, n, r, usePathSmoothing,
                   shouldBenchmark):

        routes = []

        samples = np.random.uniform(low=self.minBounds, high=self.maxBounds,
                                    size=(n, len(self.minBounds)))

        # put them in a K-D tree to allow for easy connectivity queries
        kdTree = spatial.cKDTree(samples)

        # add all start, goal, and sampled nodes
        if not shouldBenchmark:
            print('Initializing PRM...')

        graph = Graph()


        # start_node_idx = n + 1
        startState = np.asarray(startState)
        goalState = np.asarray(goalState)
        graph.add_node(n+1,
                       heuristic=self.admissible_heuristic_dist(startState, goalState),
                       prev=None, dist=0, priority=0, pos=startState.flatten())

        # goal_node_idx = n + 2
        graph.add_node(n+2,
                       heuristic=0, prev=None, dist=np.inf,
                       priority=np.inf, pos=goalState.flatten())

        # now initialize the sampled nodes of the underlying PRM graph
        for sample in range(0, n):

            pos = samples[sample, :]
            heuristic = self.admissible_heuristic_dist(pos, goalState)
            graph.add_node(sample,
                           heuristic=heuristic,
                           prev=None, dist=np.inf,
                           priority=np.inf, pos=pos.flatten())

        (graph, start_node_idx, goal_node_idx) =  (graph, n+1, n+2)

        # now connect all of the samples within radius r of each other
        if not shouldBenchmark:
            print('Connecting PRM...')

        # keep a union-find data structure to improve search performance by not
        # allowing cycles in the graph
        split_graph = UnionFind()

        for curr_node_index, curr_node_dat in list(graph.nodes(data=True)):

            curr_pos = curr_node_dat['pos']

            # search for all nodes in radius of the current node in question
            nbrs = kdTree.query_ball_point(curr_pos.flatten(),r)

            # adding all NEW edges that don't collide to the graph
            for nbrIndex in nbrs:

                gaol_xy = graph.getter_helper(nbrIndex, 'pos')

                collides = self.check_for_obs_collission(curr_pos, gaol_xy)
                check_comp = self.checkConnectivity(split_graph,
                                                            curr_node_index,
                                                            nbrIndex)

                if (not collides) and check_comp:

                    weight = self.admissible_heuristic_dist(curr_pos, gaol_xy)
                    graph.add_edge(curr_node_index, nbrIndex,
                                   weight=weight)

                    # need to update union-find data with the new edge
                    split_graph.union(curr_node_index, nbrIndex)

        if not shouldBenchmark:
            print('Finding path through PRM...')

        (shortestPath,
         pathLength, _) = graph.get_path(start_node_idx,
                                               goal_node_idx,
                                               algo='A star')
        foundPath = (shortestPath is not None)

        # only start smoothing if desired
        if foundPath and usePathSmoothing:

            if not shouldBenchmark:
                print('Smoothing path found through PRM...')

            (shortestPath,
             pathLength) = self.smoothPathInGraph(graph, shortestPath,
                                                  goal_node_idx, pathLength,
                                                  shouldBenchmark)

        # run robot through whole path
        if foundPath:
            for node in shortestPath:

                currPos = graph.getter_helper(node, 'pos')
                routes.append(currPos)
                # self.robot.updateRobotState(currPos)

        return (graph, shortestPath, pathLength, foundPath, routes)

    def findPathToGoal(self, startState, goalState, plannerConfigData,
                       plotConfigData, shouldBenchmark):

        # # allow the user to overide the settings in the config file
        plannerConfigData = None

        # n = self.n[0]
        # r = self.r[0]
        # usePathSmoothing = self.usePathSmoothing[0]
        n = self.n
        r = self.r
        usePathSmoothing = self.usePathSmoothing

        start = timer()
        (graph,
         shortestPath,
         pathLength, foundPath, routes) = self.computePRM(startState, goalState, n, r,
                                                  usePathSmoothing,
                                                  shouldBenchmark)
        finish = timer()
        computationTime = finish - start

        # plot the resulting path over the PRM computation
        shouldPlot = plotConfigData['shouldPlot']

        if(pathLength == None):
            pathLength = 0
            computationTime = 0

        if shouldPlot:
            if not pathLength:
                pathLength = np.nan
            title = 'PRM - path length = %0.3g  n = %0.3g  r = %0.3g' \
                % (pathLength, n, r)
            plotConfigData['plotTitle'] += title
            self.plot(graph, startState, goalState, plotConfigData,
                      routes, path=shortestPath)

        # print("Path Length" , pathLength)

        return (computationTime, pathLength, foundPath)

    def plot(self, graph, startState, goalState,
             plotConfigData, routes, path=None):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # plot the graph and its shortest path
        fig, ax = graph.plot(path=path, fig=fig, showLabels=False,
                             showEdgeWeights=False)

        # unpack dictionary
        plotTitle = plotConfigData['plotTitle']
        xlabel = plotConfigData['xlabel']
        ylabel = plotConfigData['ylabel']
        shouldPlotCSpaceDiscretizationGrid = False
        shouldPlotObstacles = True


        # plot grid lines BEHIND the data
        ax.set_axisbelow(True)

        plt.grid()

        # plotting all the obstacles
        if shouldPlotObstacles:
            obstacles = WORKSPACE_CONFIG['WO1']
            for obst in obstacles:
                x,y = obst.exterior.xy
                ax.fill(x,y, alpha=0.5, fc='k',ec='none')

        # plotting the robot's motion
        # if robot is not None:
        robotPath = routes
        # robotPath = path

        # plotting the robot origin's path through cspace
        x = [state[0] for state in robotPath]
        y = [state[1] for state in robotPath]
        plt.plot(x, y, color='blue', marker='*', linestyle='none',
                    linewidth=4, markersize=3,
                    label='Robot path')

        # plotting the start / end location of the robot
        plt.plot(startState[0], startState[1],
                 color='green', marker='o', linestyle='none',
                 linewidth=2, markersize=16,
                 label='Starting State')

        plt.plot(goalState[0], goalState[1],
                 color='red', marker='x', linestyle='none',
                 linewidth=4, markersize=16,
                 label='Goal State')

        ax.set_aspect('equal')
        plt.title(plotTitle)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
        ax.set_xlim(self.minBounds[0], self.maxBounds[0])
        ax.set_ylim(self.minBounds[1], self.maxBounds[1])
        fig.legend(loc='upper left')


def savePlot(fig, shouldSavePlots, baseSaveFName, plotTitle,
             useTightLayout=True):
    print("Saving fig: ", plotTitle)

    if shouldSavePlots:
        saveFName = baseSaveFName + '-' + plotTitle + '.png'
        if useTightLayout:
            plt.tight_layout()
        plt.savefig(saveFName, dpi=500)

        print('wrote figure to: ', saveFName)
        # plt.show()
        plt.close(fig)

def plotStatistics(benchMarkingDF, pathValidityDF, benchParams, baseSaveFName, plotTitle):

    print("Entering Plotting Stastics")
    ##
    # Plotting boxplots
    ##
    boxPlotsToMake = ['computationTimeInSeconds', 'pathLength']

    # need to create a new, merged categorical data for boxplots
    mergedParamsName = ', '.join(benchParams)
    benchMarkingDF[mergedParamsName] = benchMarkingDF[benchParams].apply(
        lambda x: ', '.join(x.astype(str)), axis=1)
    pathValidityDF[mergedParamsName] = pathValidityDF[
        benchParams].apply(lambda x: ', '.join(x.astype(str)), axis=1)

    # Usual boxplot for each variable that was benchmarked
    for plotVar in boxPlotsToMake:

        # make it wider for the insanse length of xticklabels
        fig = plt.figure(figsize=(20, 10))

        plt.style.use("seaborn-darkgrid")
        bp = sns.boxplot(data=benchMarkingDF,
                            x=mergedParamsName, y=plotVar)
        sns.swarmplot(x=mergedParamsName, y=plotVar, data=benchMarkingDF,
                        color="grey")

        # for readability of axis labels
        bp.set_xticklabels(bp.get_xticklabels(), rotation=45, ha='right')

        newPlotTitle = plotVar + '-' + plotTitle
        plt.title('Benchmarking of Sampled Planner ' + plotVar)
        savePlot(fig=fig, shouldSavePlots=True,
                    baseSaveFName=baseSaveFName, plotTitle=newPlotTitle)

    # number of times a valid path was found
    fig = plt.figure()

    plt.style.use('seaborn-darkgrid')
    bp = sns.barplot(x=mergedParamsName, y='numValidPaths',
                        data=pathValidityDF)
    plt.title('Number of Valid Paths Found for Each Parameter Combination')

    # for readability of axis labels
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45, ha='right')

    newPlotTitle = 'numPaths' + '-' + plotTitle
    savePlot(fig=fig, shouldSavePlots=True,
                baseSaveFName=baseSaveFName, plotTitle=newPlotTitle)

def main():
    print(" Config Files need to be updated for different results ")
    val = input(" Enter 0 for Defualt Single setup or 1 for Benchmarking: ")
    val = int(val)

    if val == 0:
        name = 'prm_w01_backup.yaml'
        with open('prm_w01_backup.yaml', 'r') as stream:
                configData = yaml.load(stream, Loader=yaml.Loader)
        
        # prm = PRM()
        
        numRunsOfPlannerPerSetting = configData['numRunsOfPlannerPerSetting']
        parametersToVary = configData['paramterNamesToVary']
        allParams = dict((var, configData[var]) for var in parametersToVary)
        print(allParams)

        keys, values = zip(*allParams.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(experiments)
        data = []
        pathValidityData = []

        print("Running Experimenents")
        print(experiments)

        for experiment in experiments:
            print("Currently Running Experiment")
            print(experiment)
            prm = None
            prm = PRM(name=name, n=experiment['n'], r=experiment['r'], smoothing=experiment['smoothing'])
            plotConfigData = {'shouldPlot': True,
                            'plotTitle': '',
                            'xlabel': 'x',
                            'ylabel': 'y',
                            'plotObstacles': True,
                            'plotGrid': False}
            print(experiment)
            numValidPaths = 1
            runInfo = {}

            for idx, i in enumerate(range(0, numRunsOfPlannerPerSetting)):
                print(idx)
                (computationTime,
                pathLength,
                fp) = prm.findPathToGoal(startState=configData['startState'],
                                            goalState=configData['goalState'],
                                            plotConfigData=plotConfigData,
                                            plannerConfigData=experiment,
                                            shouldBenchmark=True)

        plt.show()

    elif val == 1:
        name = 'prm_w01.yaml'
        with open('prm_w01.yaml', 'r') as stream:
                configData = yaml.load(stream, Loader=yaml.Loader)
        
        # prm = PRM()
        
        numRunsOfPlannerPerSetting = configData['numRunsOfPlannerPerSetting']
        parametersToVary = configData['paramterNamesToVary']
        allParams = dict((var, configData[var]) for var in parametersToVary)
        print(allParams)

        keys, values = zip(*allParams.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(experiments)
        data = []
        pathValidityData = []

        print("Running Experimenents")
        print(experiments)

        for experiment in experiments:
            print("Currently Running Experiment")
            print(experiment)
            prm = None
            prm = PRM(name=name, n=experiment['n'], r=experiment['r'], smoothing=experiment['smoothing'])
            plotConfigData = {'shouldPlot': True,
                            'plotTitle': '',
                            'xlabel': 'x',
                            'ylabel': 'y',
                            'plotObstacles': True,
                            'plotGrid': False}
            print(experiment)
            numValidPaths = 1
            runInfo = {}

            for idx, i in enumerate(range(0, numRunsOfPlannerPerSetting)):
                print(idx)
                (computationTime,
                pathLength,
                fp) = prm.findPathToGoal(startState=configData['startState'],
                                            goalState=configData['goalState'],
                                            plotConfigData=plotConfigData,
                                            plannerConfigData=experiment,
                                            shouldBenchmark=True)

                # dat = None
                dat = {'computationTimeInSeconds': computationTime, 'pathLength': pathLength}
                # bencmarkingInfo = None 
                # fp = None
                bencmarkingInfo = {**dat, **experiment}
                # benchmarkingInfo = None 
                # foundPath = None 
                (benchmarkingInfo, foundPath) = (bencmarkingInfo, fp)
                print(foundPath)
            
                benchmarkingInfo.update(experiment)
                data.append(benchmarkingInfo)

                # print(foundPath)

                if foundPath:
                    numValidPaths += 1


                
            runInfo['numValidPaths'] = copy.deepcopy(numValidPaths)
            runInfo['numTimesRun'] = numRunsOfPlannerPerSetting
            runInfo.update(copy.deepcopy(experiment))
            pathValidityData.append(runInfo)

            print(runInfo)


        benchMarkingDF = pd.DataFrame(data)
        pathValidityDF = pd.DataFrame(pathValidityData)

        benchMarkingDF.to_csv('/home/arpit/studies/motion/Assignment4/benchMarkingDF.csv',header=True)
        pathValidityDF.to_csv('/home/arpit/studies/motion/Assignment4/pathValidityDF.csv',header=True)

        (benchMarkingDF, pathValidityDF, benchParams) = (benchMarkingDF, pathValidityDF, parametersToVary)

        plotTitle = 'PRM' + '_stats'

        my_path = os.path.abspath(__file__) + '\plots'

        plotStatistics(benchMarkingDF=benchMarkingDF,
                        pathValidityDF=pathValidityDF,
                        benchParams=benchParams,
                        baseSaveFName=my_path,
                        plotTitle=plotTitle)




if __name__ == '__main__':
    main()