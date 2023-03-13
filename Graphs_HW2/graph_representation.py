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
        self._number_of_edges = number_of_edges
        self._dictionary_in = {}
        self._dictionary_out = {}
        self._dictionary_cost = {}
        for index in range(number_of_vertices):
            self._dictionary_in[index] = []
            self._dictionary_out[index] = []

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
        if x in self._dictionary_in.keys() and x in self._dictionary_out.keys():
            return False
        self._dictionary_in[x] = []
        self._dictionary_out[x] = []
        self._number_of_vertices += 1
        return True

    def remove_vertex(self, x):
        if x not in self._dictionary_in.keys() and x not in self._dictionary_out.keys():
            return False
        self._dictionary_in.pop(x)
        self._dictionary_out.pop(x)
        for key in self._dictionary_in.keys():
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
        if x not in self._dictionary_in.keys():
            return -1
        return len(self._dictionary_in[x])

    def out_degree(self, x):
        if x not in self._dictionary_out.keys():
            return -1
        return len(self._dictionary_out[x])

    def add_edge(self, x, y, cost):
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
        if x not in self._dictionary_in.keys() or y not in self._dictionary_in.keys() or x not in self._dictionary_out.keys() or y not in self._dictionary_out.keys():
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
    file = open(filename, "r")
    line = file.readline()
    line = line.strip()
    vertices, edges = line.split(' ')
    graph = TripleDictGraph(int(vertices), int(edges))
    line = file.readline().strip()
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

