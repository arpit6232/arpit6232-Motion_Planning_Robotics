import math
import random

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt 
import numpy as np 
from scipy import spatial 
# https://www.geeksforgeeks.org/union-find/
from networkx.utils import UnionFind 
from shapely.geometry import Point, LineString, Polygon, \
    MultiPolygon
import random
import copy 
# from graph import Graph
from workspaces import config
import yaml
import itertools
from timeit import default_timer as timer
import pandas as pd 
import seaborn as sns
import os
import random
import math

WORKSPACE_CONFIG = config()

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self):

        name = 'rrt_WO1.yaml'
        with open(name, 'r') as stream:
            configData = yaml.load(stream, Loader=yaml.Loader)

        self.startState = configData['startState']
        self.goalState = configData['goalState']
        self.minBounds = configData['minBounds']
        self.maxBounds = configData['maxBounds']

        self.start = self.Node(configData['startState'][0], configData['startState'][1])
        self.end = self.Node(configData['goalState'][0], configData['goalState'][1])
        self.radius = configData['radius']
        self.grid_size = configData['grid_size']
        self.goal_sample_rate = configData['goal_sample_rate']
        self.max_iter = configData['max_iter']
        self.graph = []

    def planning(self, animation=True):
        pathLength = 0
        computationTime = 0
        start = timer()
        path_generated = False
        self.graph = [self.start]
        for i in range(self.max_iter):
            sample = self.get_random_node()

            inter_node_dist_list = [(node.x - sample.x)**2 + (node.y - sample.y)**2 for node in self.graph]
            temp = inter_node_dist_list.index(min(inter_node_dist_list))
            next_node = self.graph[temp]

            upgraded_node = self.directional_growth(next_node, sample, self.radius)

            if self.check_collision(upgraded_node):
                self.graph.append(upgraded_node)

            if animation and i % 5 == 0:
                self.plotGraph(sample)

            if self.GAOLdist(self.graph[-1].x, self.graph[-1].y) <= self.radius:
                final_node = self.directional_growth(self.graph[-1], self.end, self.radius)
                if self.check_collision(final_node):
                    _path = [[self.end.x, self.end.y]]
                    _node = self.graph[(len(self.graph) - 1)]
                    while _node.parent is not None:
                        _path.append([_node.x, _node.y])
                        _node = _node.parent
                    _path.append([_node.x, _node.y])
                    path = _path

            if animation and i % 5:
                self.plotGraph(sample)

        finish = timer()
        computationTime = finish - start

        if path is not None:
            start_x = 0
            start_y = 0
            pathLength = 0
            path_generated = True
            for (x, y) in path:
                dx, dy = (start_x - x), (start_y - y)
                d = math.hypot(dx, dy)
                start_x = x
                start_y = y
                pathLength += d

            return (computationTime, path, pathLength, True)

        return (None, None, None, False)  # cannot find path

    def directional_growth(self, src, dest, grow_len=float("inf")):

        upgraded_node = self.Node(src.x, src.y)
        dx = (dest.x - upgraded_node.x)
        dy = (dest.y - upgraded_node.y)
        euclid_dist = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)

        upgraded_node.path_x, upgraded_node.path_y = [upgraded_node.x], [upgraded_node.y]

        if grow_len > euclid_dist:
            grow_len = euclid_dist

        for _ in range(math.floor(grow_len / self.grid_size)):
            upgraded_node.x += self.grid_size * math.cos(theta)
            upgraded_node.y += self.grid_size * math.sin(theta)
            upgraded_node.path_x.append(upgraded_node.x)
            upgraded_node.path_y.append(upgraded_node.y)

        dx, dy = (dest.x - upgraded_node.x), (dest.y - upgraded_node.y)
        # dy = dest.y - upgraded_node.y
        euclid_dist = math.hypot(dx, dy)
        if euclid_dist <= self.grid_size:
            upgraded_node.path_x.append(dest.x)
            upgraded_node.path_y.append(dest.y)
            upgraded_node.x = dest.x
            upgraded_node.y = dest.y

        upgraded_node.parent = src

        return upgraded_node

    def GAOLdist(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.minBounds[0], self.maxBounds[0]),
                random.uniform(self.minBounds[1], self.maxBounds[1]))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def plotGraph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.graph:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        obstacles = WORKSPACE_CONFIG['WO1']
        for obst in obstacles:
            x,y = obst.exterior.xy
            plt.fill(x,y, alpha=0.5, fc='c',ec='none')

        plt.plot(self.startState[0], self.startState[1], "xr")
        plt.plot(self.goalState[0], self.goalState[1], "xr")
        plt.axis("equal")
        plt.axis([self.minBounds[0], self.maxBounds[0], self.minBounds[1], self.maxBounds[1]])
        plt.grid(True)
        # plt.pause(0.01)

    def check_collision(self, node):

        obstacles = WORKSPACE_CONFIG['WO1']
        p = Point(node.x, node.y)
        collide_flag = False
        for obstacle in obstacles:
            if p.within(obstacle):

                collide_flag = True
                break

        if collide_flag:
            return False
        else:
            return True


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

    val = input("Enter 0 to get a single run, Enter 1 for Benchmarking Plot: ")
    val = int(val)

    if val == 0:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axisbelow(True)

        name = 'rrt_WO1.yaml'
        with open(name, 'r') as stream:
                configData = yaml.load(stream, Loader=yaml.Loader)

        minBounds = configData['minBounds']
        maxBounds = configData['maxBounds']

        distance = 0

        numRunsOfPlannerPerSetting = 100
        parametersToVary = configData['paramterNamesToVary']
        allParams = dict((var, configData[var]) for var in parametersToVary)
        print(allParams)

        keys, values = zip(*allParams.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(experiments)
        data = []
        pathValidityData = []

        plotConfigData = {'shouldPlot': True,
                            'plotTitle': '',
                            'xlabel': 'x',
                            'ylabel': 'y',
                            'plotObstacles': True,
                            'plotGrid': False}

        rrt = RRT()

        (computationTime, path, pathLength, path_generated) = rrt.planning(animation=False)

        if path is None:
            print("Algorithm convergence failed in the specified number of iterations")
        else:
            print("You Bet: GOT A PATH")
            # # Draw final path
            if show_animation:
                rrt.plotGraph()
                plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                plt.grid(True)
        
        start_x = 0
        start_y = 0
        pathLength = 0

        for (x, y) in path:
            dx = start_x - x
            dy = start_y - y
            d = math.hypot(dx, dy)
            start_x = x
            start_y = y
            pathLength += d

        r = 0.5

        plotTitle = 'RRT - path length = %0.3g r = %0.3g computationTime = %0.3g  '\
             % (pathLength/2, r, computationTime/100)
        
        ax.set_aspect('equal')
        plt.title(plotTitle)
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
        ax.set_xlim(minBounds[0], maxBounds[0])
        ax.set_ylim(minBounds[1], maxBounds[1])
        fig.legend(loc='upper left')

        plt.show()

    if val == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axisbelow(True)

        name = 'rrt_WO1.yaml'
        with open(name, 'r') as stream:
                configData = yaml.load(stream, Loader=yaml.Loader)

        minBounds = configData['minBounds']
        maxBounds = configData['maxBounds']

        distance = 0

        numRunsOfPlannerPerSetting = 100
        parametersToVary = configData['paramterNamesToVary']
        allParams = dict((var, configData[var]) for var in parametersToVary)

        keys, values = zip(*allParams.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(experiments)
        data = []
        pathValidityData = []

        plotConfigData = {'shouldPlot': True,
                            'plotTitle': '',
                            'xlabel': 'x',
                            'ylabel': 'y',
                            'plotObstacles': True,
                            'plotGrid': False}

        for experiment in experiments:
            rrt = None
            rrt = RRT()
            numValidPaths = 1
            runInfo = {}

            for idx, i in enumerate(range(0, 100)):
                (computationTime, path, pathLength, path_generated) = rrt.planning(animation=False)

                dat = {'computationTimeInSeconds': computationTime, 'pathLength': pathLength}
                bencmarkingInfo = {**dat, **experiment}
                (benchmarkingInfo, path_generated) = (bencmarkingInfo, path_generated)

                benchmarkingInfo.update(experiment)
                data.append(benchmarkingInfo)

                if path_generated:
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