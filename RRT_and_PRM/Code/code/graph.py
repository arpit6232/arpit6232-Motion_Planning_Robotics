"""
Arpit Savarkar
Implemenatation of Graph Search 
for Dijkstra and A*

"""

import matplotlib.pyplot as plt
import yaml
import networkx as nx
import heapq
import queue
import numpy as np
import copy


class Graph(nx.Graph):
    """
    Graph Class for handy functions
    """

    def __init__(self, nodes=[], edges=[]):

        # networkx initialization is also required 
        super().__init__()

        if nodes:
            self.add_nodes_from(nodes)

            if edges:
                self.add_edges_from(edges)

            self.nodeProperties = set([k for n in self.nodes
                                       for k in self.nodes[n].keys()])

    def get_path(self, start, goal, algo):
        """
        Returns the Path From the nodes based on Graph 
        PARAMETERS
        ----------
        start : Start Node 
        goal : Goal Node 
        algo : String - "A star" or "Dijkstra"
        
        RETURNS
        -------
        path: A list of the nodes 
        pathLength: Integer - Length of the Algorithamic plan 
        itrs: Number of iterations it took to find the result 
        """
        path = None
        pathLength = None
        itrs = 0

        if start == goal:
            path = [start]
            return (path, pathLength, itrs)

        # Node is the structure that holds the priority, prev node and the distance
        for node in self.nodes:

            self.setter_helper(node, 'distance', np.inf)
            self.setter_helper(node, 'priority', np.inf)
            self.setter_helper(node, 'prev', None)

        # Sets the hueristic distance to 0 for the start node 
        self.setter_helper(start, 'priority', 0)
        self.setter_helper(start, 'distance', 0)

        # Priority Queue implementation 
        Q = PrioQueue()

        # Pushes onto the FIFO(Queue)
        Q.put(self.getPriorityTuple(start))

        while not Q.empty():

            currPriority, currNode = Q.get()
            itrs += 1
            
            # Success
            if self.searchStop(currNode, goal, algo):

                pathLength = self.getter_helper(currNode, 'distance')
                path = self.revv(start, goal)
                break
            
            # Checks the Neighhbour for the heuristic with weight additions
            for neighbor in self.adj[currNode]:

                edg = (currNode, neighbor)

                src = edg[0]
                dest = edg[1]
                weight = self.edges[edg]['weight']

                srcDist = self.getter_helper(src, 'distance')
                srcPriority = self.getter_helper(src, 'priority')

                distance_to_label_dest = self.getter_helper(dest, 'distance')
                destPriority = self.getter_helper(dest, 'priority')

                # Constraint Condition
                if not (self.getter_helper(src, 'prev') == dest):
                    flag = (srcDist + weight) < destPriority
                else:
                    flag = False
                
                if flag:
                    # Updates the path accordingly, push onto the priority_queue
                    self.update_to_shorter_route(edg, algo)
                    Q.put(self.getPriorityTuple(neighbor))

        return (path, pathLength, itrs)


    def update_to_shorter_route(self, edgeLabel, algo):
        """
        Update step to find a route shorter than the existing calculated route 
        PARAMETERS
        ----------
        edgeLabel: Edge Index
        algo: String - "A star" or "Dijkstra"

        RETURNS
        -------
        distance_to_label_dest: Distance Count
        """
        src = edgeLabel[0]
        dest = edgeLabel[1]
        weight = self.edges[edgeLabel]['weight']

        srcDist = self.getter_helper(src, 'distance')
        srcPriority = self.getter_helper(src, 'priority')

        distance_to_label_dest = self.getter_helper(dest, 'distance')
        destPriority = self.getter_helper(dest, 'priority')

        if algo == 'A star':

            distance_to_label_dest = copy.deepcopy(self.getter_helper(src, 'distance')) + weight
            destPriority = copy.deepcopy(distance_to_label_dest) + \
                self.getter_helper(dest, 'heuristic')

        elif algo == 'Dijkstra':

            distance_to_label_dest = copy.deepcopy(self.getter_helper(src, 'distance')) + weight
            destPriority = copy.deepcopy(distance_to_label_dest)


        self.setter_helper(dest, 'distance', distance_to_label_dest)
        self.setter_helper(dest, 'priority', destPriority)

        if not (self.getter_helper(src, 'prev') == dest):
            self.setter_helper(dest, 'prev', src)

        return distance_to_label_dest

    def searchStop(self, currNode, goal, algo):
        """
        Success Condition 
        PARAMETERS
        ----------
        currNode : Node under consideration 
        goal: Goal Node 
        algo : String - "A star" or "Dijkstra"

        RETURNS
        -------
        atGoal: Boolean . True/False
        """

        atGoal = (currNode == goal)

        if algo == 'A star':

            if atGoal:

                prevNode = self.getter_helper(currNode, 'prev')
                currDist = self.getter_helper(currNode, 'distance')
                prevPriority = self.getter_helper(prevNode, 'priority')

                return (currDist <= prevPriority)

        elif algo == 'Dijkstra':

            return atGoal

    def getPathEdges(self, path):
        """
        PARAMETERS
        ----------
        path: List of Nodes

        RETURNS
        -------
        path_edges: Path_Edges 
        """

        path_edges = [(v1, v2) for v1, v2 in zip(path, path[1:])]

        return path_edges

    def revv(self, start, goal):
        """
        Reverse the Path, gives if found from start to goal
        PARAMETERS
        ---------
        start: Start Node 
        goal: Goal Node
        """

        currNode = goal
        path = [currNode]

        while currNode != start:

            currNode = self.getter_helper(currNode, 'prev')
            path.append(currNode)

        path.reverse()

        return path

    def getPriorityTuple(self, node):
        """
        Returns the priority 
        PARAMETERS
        ----------
        Node : Node
        """

        return (self.getter_helper(node, 'priority'), node)

    def getter_helper(self, nodeLabel, dataKey):
        """
        GETTER FUNCTION 

        PARAMETERS
        ---------
        nodeLabel: Node Index 
        dataKey: 
        """

        nodeData = self.nodes.data()

        return nodeData[nodeLabel][dataKey]

    def setter_helper(self, nodeLabel, dataKey, data):
        """ 
        SETTER Function 

        PARAMETERS
        ----------
        nodeLabel: Node Index
        data: Dataset to set
        dataKey:

        """

        nodeData = self.nodes.data()
        nodeData[nodeLabel][dataKey] = data

    def print_edges(self):
        """
        Prints the Edges 

        PARAMETERS
        ----------
        Segmented Print statement
        """

        for n, nbrs in self.adj.items():
            for nbr, eattr in nbrs.items():

                wt = eattr['weight']
                print('(%s, %s, %0.3g)' % (str(n), str(nbr), wt))

    def dispNodes(self):
        """
        Prints Node in the graph established
        """
        for node in self.nodes(data=True):
            print(node)

    def plot(self, path=None, fig=None, plotTitle=None,
             baseSize=400, node_size=10, showLabels=True,
             showEdgeWeights=True, showAxes=True):
        """
        Plotting Function 
        PARAMETERS
        ----------
        path : List of nodes 
        fig: plt.figure()
        Title: Plot title 
        baseSize: Size of the figure
        node_size: Size of the nodes to be displayed (scaled)
        showLabels: Boolean 

        """
        if not fig:
            fig = plt.figure()

        # scale node sizes by string length only if all node labels are strings
        allStrs = bool(self.nodes()) and all(isinstance(elem, str)
                                             for elem in self.nodes())
        pos = nx.get_node_attributes(self, 'pos')
        if allStrs:
            node_size = [len(v) * baseSize for v in self.nodes()]
            nx.draw_networkx(self, pos=pos,
                             with_labels=showLabels, node_size=node_size)
        else:
            nx.draw_networkx(self, pos=pos, with_labels=showLabels,
                             node_size=node_size,
                             cmap=plt.get_cmap('jet'))

        # show edge weights as well
        if showEdgeWeights:
            labels = nx.get_edge_attributes(self, 'weight')
            nx.draw_networkx_edge_labels(self, pos, edge_labels=labels)

        # draw path through the graph if it exists
        if path:

            nx.draw_networkx_nodes(self, pos, nodelist=path, node_color='m',
                                   node_size=node_size)

            path_edges = self.getPathEdges(path)
            nx.draw_networkx_edges(self, pos, edgelist=path_edges,
                                   edge_color='m', width=4)

        # Axes settings
        ax = plt.gca()
        ax.set_title(plotTitle)
        if showAxes:
            ax.tick_params(left=True, bottom=True,
                           labelleft=True, labelbottom=True)
        else:
            [sp.set_visible(False) for sp in ax.spines.values()]
            ax.set_xticks([])
            ax.set_yticks([])

        return (fig, ax)

# https://stackoverflow.com/questions/5997189/how-can-i-make-a-unique-value-priority-queue-in-python

class PrioQueue(queue.Queue):
    """
    Class for Setting up the priority queue
    """

    def _init(self, maxsize):
        self.queue = []
        self.REMOVED = '<removed-task>'
        self.entry_finder = {}

    def _put(self, item, heappush=heapq.heappush):
        item = list(item)
        priority, task = item

        if task in self.entry_finder:
            # Do not add new item.
            pass
        else:
            self.entry_finder[task] = item
            heappush(self.queue, item)

    def _qsize(self, len=len):

        return len(self.entry_finder)

    def is_empty(self):
        while self.pq:
            if self.queue[0][1] != self.REMOVED:
                return False
            else:
                _, _, element = heapq.heappop(self.pq)
                if element in self.element_finder:
                    del self.element_finder[element]
        return True

    def _get(self, heappop=heapq.heappop):
        while self.queue:

            item = heappop(self.queue)
            _, task = item

            if task is not self.REMOVED:

                del self.entry_finder[task]
                return item




def print_helper(algo, pathLength, itr):
    print('|********  ', algo, '  *********|')
    if pathLength:
        print('Path Length:', pathLength)
    else:
        print('No Path Exists')
    print("Iterations: ", itr)

def plotPath(algo, graph, path, pathLength, itr, flag, flag2):
    print("**********SHORTEST PATH************")
    print(path)
    if path:
        plotTitle = 'Shortest Path (length = ' + str(pathLength) + \
                            ') Found with ' + algo + ' - nIter: ' + \
                            str(itr)
    else:
        plotTitle = 'No Path Found with ' + algo + ' - nIter: ' + \
                            str(itr)


def main():
    with open('graphConfig.yaml', 'r') as stream:
            configData = yaml.load(stream, Loader=yaml.Loader)

    nodes = configData['nodes']
    adjList = configData['edges']
    startNode = configData['startLabel']
    goalNode = configData['goalLabel']
    
    edgeList = []
    for srcEdge, dat in adjList.items():
        if dat:
            newEdges = [(srcEdge, data[0], data[1]) for data in dat]
            edgeList.extend(newEdges)
    
    G = Graph(nodes.items(), edges=edgeList)
    
    path = {}
    pathLength = {}
    num_itrs = {}

    val = input("Enter 0 for A_star and 1 for Dijkstra: ")
    val = int(val)

    if val == 0:
        algo = 'A star'
        (path[algo], 
        pathLength[algo],
        num_itrs[algo]) = G.get_path(start = startNode,
                                        goal = goalNode,
                                        algo = algo)

        print_helper(algo, pathLength[algo], num_itrs[algo])
        plotPath(algo, G, path[algo], pathLength[algo],
                    num_itrs[algo], True, True)
    elif val == 1:

        algo = 'Dijkstra'
        (path[algo],
            pathLength[algo],
            num_itrs[algo]) = G.get_path(start=startNode,
                                                goal=goalNode,
                                                algo=algo)
                            
        print_helper(algo, pathLength[algo], num_itrs[algo])
        plotPath(algo, G, path[algo], pathLength[algo],
                    num_itrs[algo], True, True)


if __name__ == '__main__':
    main()