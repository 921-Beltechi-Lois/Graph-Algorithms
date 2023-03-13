# def generator_fibonacci(n):
#     x_2 = 0
#     x_1 = 1
#
#     yield x_2
#     yield x_1
#
#     for x in range(2, n + 1):
#         yield x_2 + x_1
#         x_2, x_1 = x_1, x_2 + x_1
#
#
# gen_fibonacci = generator_fibonacci(10)
#
# print(gen_fibonacci)


import copy
import math


class TripleDictGraph:
    def __init__(self, number_of_vertices, number_of_edges):
        self._number_of_vertices = number_of_vertices
        self._vertices = []
        self._number_of_edges = number_of_edges
        self._dictionary_in = {}
        self._dictionary_out = {}
        self._dictionary_cost = {}
        for index in range(number_of_vertices):
            self._dictionary_in[index] = []
            self._dictionary_out[index] = []

    @property
    def vertices(self):
        return self._vertices

    @property
    def dictionary_cost(self):
        return self._dictionary_cost

    @property
    def dictionary_in(self):
        return self._dictionary_in

    @property
    def dictionary_out(self):
        return self._dictionary_out

    @property
    def number_of_vertices(self):
        return self._number_of_vertices

    @property
    def number_of_edges(self):
        return self._number_of_edges

    def parse_vertices(self):
        vertices = list(self._dictionary_in.keys())
        for v in vertices:
            yield v

    def parse_inbound(self, x):
        for y in self._dictionary_in[x]:
            yield y

    def parse_outbound(self, x):
        for y in self._dictionary_out[x]:
            yield y

    def parse_cost(self):
        keys = list(self._dictionary_cost.keys())
        for key in keys:
            yield key

    def add_vertex(self, x):
        """
        Add a vertex to the graph
        :param x: the vertex to be added
        :return: True if it is added, False otherwise
        """
        if x in self._vertices:
            return False  # the vertex already exists
        self._dictionary_in[x] = []
        self._dictionary_out[x] = []
        self._vertices.append(x)
        self._number_of_vertices += 1
        return True

    def remove_vertex(self, x):
        """
        Remove a vertex from the graph
        :param x: Vertex to be removed
        :return: True if it is removed, false otherwise
        """
        if x not in self._vertices:
            return False  # the vertex does not exist
        self._dictionary_in.pop(x)
        self._dictionary_out.pop(x)
        self._vertices.remove(x)
        for key in self._vertices:
            if x in self._dictionary_in[key]:
                self._dictionary_in[key].remove(x)
            elif x in self._dictionary_out[key]:
                self._dictionary_out[key].remove(x)
        keys = list(self._dictionary_cost.keys())
        for key in keys:
            if key[0] == x or key[1] == x:
                self._dictionary_cost.pop(key)
                self._number_of_edges -= 1
        self._number_of_vertices -= 1
        return True

    def in_degree(self, x):
        """
        Gets the in degree of a vertex
        :param x: vertex
        :return: the in degree or -1 if the vertex does not exist
        """
        if x not in self._vertices:
            return -1
        return len(self._dictionary_in[x])

    def out_degree(self, x):
        """
        Gets the out degree of a vertex
        :param x: vertex
        :return: the out degree or -1 if the vertex does not exist
        """
        if x not in self._vertices:
            return -1
        return len(self._dictionary_out[x])

    def add_edge(self, x, y, cost):
        """
        Add an edge to the graph
        :param x: first vertex
        :param y: second vertex
        :param cost: the cost of the edge
        :return: True if added, False otherwise
        """
        if x not in self._vertices or y not in self.vertices:
            return False
        if x in self._dictionary_in[y]:
            return False
        elif y in self._dictionary_out[x]:
            return False
        elif (x, y) in self._dictionary_cost.keys():
            return False
        self._dictionary_in[y].append(x)
        self._dictionary_out[x].append(y)
        self._dictionary_cost[(x, y)] = cost
        self._number_of_edges += 1
        return True

    def remove_edge(self, x, y):
        """
        Remove an edge from the graph
        :param x: first vertex
        :param y: second vertex
        :return: True if removed, false otherwise
        """
        if x not in self._vertices or y not in self._vertices:
            return False
        if x not in self._dictionary_in[y]:
            return False
        elif y not in self._dictionary_out[x]:
            return False
        elif (x, y) not in self._dictionary_cost.keys():
            return False
        self._dictionary_in[y].remove(x)
        self._dictionary_out[x].remove(y)
        self._dictionary_cost.pop((x, y))
        self._number_of_edges -= 1
        return True


    def find_if_edge(self, x, y):
        if x in self._dictionary_in[y]:
            return self._dictionary_cost[(x, y)]
        elif y in self._dictionary_out[x]:
            return self._dictionary_cost[(x, y)]
        return False

    def change_cost(self, x, y, cost):
        if (x, y) not in self._dictionary_cost.keys():
            return False
        self._dictionary_cost[(x, y)] = cost
        return True

    def make_copy(self):
        return copy.deepcopy(self)

    def shortest_path(self, graph, start_vertex, end_vertex):
        '''Finds the shortest (min length) path from start_vertex to end_vertex in the graph 'graph'.
        Returns the list of vertices along the path, strating with start_vertex and ending with end_vertex.
        If start_vertex == end_vertex, it returns [start_vertex]
        If there is no path, returns an empty list.
        If there are multiple optimal paths, returns one of them.
        '''
        parent = self.bfs(graph, end_vertex)
        if len(parent) == 1 and start_vertex != end_vertex:
            return []
        current = start_vertex
        path = [start_vertex]  # add first one
        while current != end_vertex:
            current = parent[current]
            path.append(current)
        #path.reverse()
        return path

    def bfs(self, graph, end_vertex):
        '''Does a BFS parsing of grapg 'graph', starting from end_vertex.
        Returns a dictionary where the keys are the vertices accessible from end_vertex and the corresponding
        values are their parents in the BFS tree. Parent of end_vertex will be None.
        '''
        parents = {}
        queue = [end_vertex]
        index_queue = 0
        parents[end_vertex] = None

        while index_queue < len(queue):
            current = queue[index_queue]
            index_queue += 1

            for neighbour in graph.parse_outbound(current):
                if neighbour not in parents.keys():  # Parents: Key: Neigh, value: Current(Parent)
                    parents[neighbour] = current  # current = parent of neighbor
                    queue.append(neighbour)

        return parents

    # Lab 3 HW
    def isEdge(self,start,end):
        """Returns True if there is an edge from x to y, False otherwise"""
        return end in self._dictionary_out[start]

    # Lab 3 HW
    def retrieveCost(self,start,end):
        if self.isEdge(start,end):
            return self._dictionary_cost[(start,end)]

    # Lab 3 HW
    def parseKeys(self):
        """returns a copy of all the vertex keys"""
        return list(self._dictionary_out.keys())

    # Lab 3 HW
    def floydWarshall(self):
        INFINITY = math.inf
        vertices = self.number_of_vertices

        """initialize the distances matrix with infinity on every position
           and the path matrix with -1 on every position"""
        distances = [[INFINITY] * vertices for i in range(vertices)]
        paths = [[-1] * vertices for i in range(vertices)]

        """initialize the (i,i) distances with 0
           and the intersection (i,i) with i"""
        for i in range(vertices):
            distances[i][i] = 0
            paths[i][i] = i

        """add the corresponding costs to the matrix"""
        for (x, y) in self._dictionary_cost:
            distances[x][y] = self.retrieveCost(x, y)
            paths[x][y] = y

        """construct the floyd-warshall distances matrix and also the path matrix"""
        for k in range(vertices):
            for i in range(vertices):
                for j in range(vertices):
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]
                        paths[i][j] = paths[i][k]

        """return the two matrices"""

        return distances[:], paths[:]

    # def topo_sort_dfs(self, vertex, sortedList, fullyProcessed, inProcess):
    #     """
    #     Function for sorting the graph topologically.
    #     :param vertex: the current vertex of the graph
    #     :param sortedList: the list with the topological order
    #     :param fullyProcessed: vertices that are fully processed
    #     :param inProcess: vertices that are in process
    #     :return: True if the vertex is valid, False otherwise
    #     """
    #     inProcess.add(vertex)  # add to the set of processing vertices the current vertex
    #     for other_vertex in self._dictionary_in[vertex]:  # process all the inbound vertices of the current vertex
    #         if other_vertex in inProcess:
    #             return False
    #         elif other_vertex not in fullyProcessed:  # if one of the inbound vertices of the current one is not processed
    #             # process it and its inbound vertices
    #             ok = self.topo_sort_dfs(other_vertex, sortedList, fullyProcessed, inProcess)
    #             if not ok:  # if we get to a vertex that is in process again we have a loop, so the graph is not a DAG
    #                 # we return False and the algorithm stops
    #                 return False
    #     inProcess.remove(vertex)
    #     sortedList.append(vertex)  # add the processed vertex to the topological sort
    #     fullyProcessed.add(vertex)  # add the processed vertex to the set of all the processed vertices
    #     return True
    #
    # def base_topo_sort(self):
    #     """
    #     Helper function for the topological sort.
    #     :return:the topological sort of the graph
    #     """
    #     sorted_list = []  # list for the topological sort
    #     fully_processed = set()  # set for all the processed vertices
    #     in_process = set()  # set for the vertices that are in process
    #     for vertex in self._vertices:  # go through all the vertices of the graph
    #         if vertex not in fully_processed:
    #             ok = self.topo_sort_dfs(vertex, sorted_list, fully_processed, in_process)
    #             if not ok:  # the graph is not a DAG so we return an empty list; it does not have a topological sort
    #                 sorted_list.clear()
    #                 return []
    #     return sorted_list

    def toposort(self):
        counter = {}
        queue = []
        for x in self.vertices:
            counter[x] = len(list(self.parse_inbound(x)))
            if counter[x] == 0:
                queue.append(x)
        sorted_list = []
        while queue:
            x = queue.pop()
            sorted_list.append(x)
            for y in self.parse_outbound(x):
                counter[y] -= 1
                if counter[y] == 0:
                    queue.append(y)
        print(f"sorted_list={sorted_list}")
        if len(sorted_list) == len(counter):
            return sorted_list
        else:
            #print(get_cycle(graph, counter))
            return None

    def highest_cost_path(self, vertex_begin, vertex_end):
        """
        Function to compute the highest cost path from vertex begin to vertex end.
        :param vertex_begin: the beginning of the path
        :param vertex_end: the end of the path
        :return: the distance(cost) of the path and the dictionary of previous vertices
        """
        #topological_order_list = self.base_topo_sort()  # get the topological sort
        topological_order_list = self.toposort()
        dist = {}  # dictionary of costs from the source
        prev = {}  # dictionary that stores for each vertex the previous vertex from the path
        m_inf = float('-inf')
        for vertex in topological_order_list:  # initialize all the values of the dictionaries
            dist[vertex] = m_inf
            prev[vertex] = -1
        dist[vertex_begin] = 0

        for vertex in topological_order_list:  # go through all the vertices
            if vertex == vertex_end:  # stop the loop if we get to the end vertex
                break
            for other_vertex in self._dictionary_out[vertex]:  # parse the outbound vertices of the current vertex
                if dist[other_vertex] < dist[vertex] + self._dictionary_cost[(vertex, other_vertex)]:
                    # if the cost is greater update the dictionary
                    dist[other_vertex] = dist[vertex] + self._dictionary_cost[(vertex, other_vertex)]
                    prev[other_vertex] = vertex

        return dist[vertex_end], prev


def write_graph_to_file(graph, file):
    file = open(file, "w")
    first_line = str(graph.number_of_vertices) + ' ' + str(graph.number_of_edges) + '\n'
    file.write(first_line)
    if len(graph.dictionary_cost) == 0 and len(graph.dictionary_in) == 0:
        raise ValueError("There is nothing that can be written!")
    for edge in graph.dictionary_cost.keys():
        new_line = "{} {} {}\n".format(edge[0], edge[1], graph.dictionary_cost[edge])
        file.write(new_line)
    for vertex in graph.dictionary_in.keys():
        if len(graph.dictionary_in[vertex]) == 0 and len(graph.dictionary_out[vertex]) == 0:
            new_line = "{}\n".format(vertex)
            file.write(new_line)
    file.close()


def read_graph_from_file(filename):
    """
    Read a graph from a file
    :param filename: the name of the file
    :return: the graph
    """
    file = open(filename, "r")
    line = file.readline()
    line = line.strip()
    vertices, edges = line.split(' ')
    graph = TripleDictGraph(int(vertices), int(edges))
    line = file.readline().strip()
    for vertex in range(int(vertices)):
        graph.vertices.append(vertex)
    while len(line) > 0:
        line = line.split(' ')
        if len(line) == 1:
            graph.dictionary_in[int(line[0])] = []
            graph.dictionary_out[int(line[0])] = []
        else:
            graph.dictionary_in[int(line[1])].append(int(line[0]))
            graph.dictionary_out[int(line[0])].append(int(line[1]))
            graph.dictionary_cost[(int(line[0]), int(line[1]))] = int(line[2])
        line = file.readline().strip()
    file.close()
    return graph

