import math
from random import randint

from graph_representation import TripleDictGraph, write_graph_to_file, read_graph_from_file


class UI:
    def __init__(self):
        self._graphs = []
        self._current_index_of_graph_list = None

    def add_empty_graph(self):
        if self._current_index_of_graph_list is None:
            self._current_index_of_graph_list = 0
        graph = TripleDictGraph(0, 0)
        self._graphs.append(graph)
        self._current_index_of_graph_list = len(self._graphs) - 1

    def create_random_graph_ui(self):
        vertices = int(input("Input the number of vertices: "))
        edges = int(input("Input the number of edges: "))
        graph = self.generate_random(vertices, edges)
        if self._current_index_of_graph_list is None:
            self._current_index_of_graph_list = 0
        self._graphs.append(graph)
        self._current_index_of_graph_list = len(self._graphs) - 1

    def generate_random(self, vertices, edges):
        if edges > vertices * vertices:
            raise ValueError("Too many edges!")
        graph = TripleDictGraph(vertices, 0)
        i = 0
        while i < edges:
            x = randint(0, vertices - 1)
            y = randint(0, vertices - 1)
            cost = randint(0, 500)
            if graph.add_edge(x, y, cost):
                i += 1
        return graph

    def read_graph_from_file_ui(self):
        filename = input("Input the name of the file: ")
        if self._current_index_of_graph_list is None:
            self._current_index_of_graph_list = 0
        graph = read_graph_from_file(filename)
        self._graphs.append(graph)
        self._current_index_of_graph_list = len(self._graphs) - 1

    def write_graph_to_file_ui(self):
        current_graph = self._graphs[self._current_index_of_graph_list]
        output_file = "output" + str(self._current_index_of_graph_list) + ".txt"
        write_graph_to_file(current_graph, output_file)

    def switch_graph_ui(self):
        print("You are on the graph number: {}".format(self._current_index_of_graph_list))
        print("Available graphs: 0 - {}".format(str(len(self._graphs) - 1)))
        number = int(input("Input the graph number you want to switch to: "))
        if not 0 <= number < len(self._graphs):
            raise ValueError("Trying to switch to a non existing graph!")
        self._current_index_of_graph_list = number

    def get_number_of_vertices_ui(self):
        print(
            "The number of vertices is: {}.".format(self._graphs[self._current_index_of_graph_list].number_of_vertices))

    def get_number_of_edges_ui(self):
        print("The number of edges is: {}.".format(self._graphs[self._current_index_of_graph_list].number_of_edges))

    def list_all_outbound(self):
        for x in self._graphs[self._current_index_of_graph_list].parse_vertices():
            line = str(x) + " :"
            for y in self._graphs[self._current_index_of_graph_list].parse_outbound(x):
                line = line + " " + str(y)
            print(line)

    def list_outbound(self):
        vertex = int(input("Input the vertex: "))
        line = str(vertex) + " :"
        for y in self._graphs[self._current_index_of_graph_list].parse_outbound(vertex):
            line = line + " " + "({}, {})".format(str(vertex), str(y))
        print(line)

    def list_all_inbound(self):
        for x in self._graphs[self._current_index_of_graph_list].parse_vertices():
            line = str(x) + " :"
            for y in self._graphs[self._current_index_of_graph_list].parse_inbound(x):
                line = line + " " + str(y)
            print(line)

    def list_inbound(self):
        vertex = int(input("Input the vertex: "))
        line = str(vertex) + " :"
        for y in self._graphs[self._current_index_of_graph_list].parse_inbound(vertex):
            line = line + " " + "({}, {})".format(str(y), str(vertex))
        print(line)

    def list_all_costs(self):
        for key in self._graphs[self._current_index_of_graph_list].parse_cost():
            line = "({}, {})".format(key[0], key[1]) + " :" + str(
                self._graphs[self._current_index_of_graph_list].dictionary_cost[key])
            print(line)

    def parse_all_vertices(self):
        for vertex in self._graphs[self._current_index_of_graph_list].parse_vertices():
            print("{}".format(vertex))

    def add_vertex_ui(self):
        vertex = int(input("Input the new vertex: "))
        added = self._graphs[self._current_index_of_graph_list].add_vertex(vertex)
        if added:
            print("Vertex added successfully!")
        else:
            print("Cannot add this vertex, it already exists!")

    def delete_vertex_ui(self):
        vertex = int(input("Enter the vertex to be deleted: "))
        deleted = self._graphs[self._current_index_of_graph_list].remove_vertex(vertex)
        if deleted:
            print("Vertex deleted successfully!")
        else:
            print("Cannot delete this vertex, it does not exist!")

    def add_edge_ui(self):
        print("Add an edge (an edge is (x, y))")
        vertex_x = int(input("Enter x = "))
        vertex_y = int(input("Enter y = "))
        cost = int(input("Enter the cost of the edge: "))
        added = self._graphs[self._current_index_of_graph_list].add_edge(vertex_x, vertex_y, cost)
        if added:
            print("Edge added successfully!")
        else:
            print("Cannot add this edge, it already exists!")

    def remove_edge_ui(self):
        print("Remove an edge (an edge is (x, y))")
        vertex_x = int(input("Enter x = "))
        vertex_y = int(input("Enter y = "))
        deleted = self._graphs[self._current_index_of_graph_list].remove_edge(vertex_x, vertex_y)
        if deleted:
            print("Edge deleted successfully!")
        else:
            print("Cannot remove this edge, it does not exist!")

    def modify_cost_ui(self):
        print("Modify the cost of an edge (an edge is (x, y))")
        vertex_x = int(input("Enter x = "))
        vertex_y = int(input("Enter y = "))
        cost = int(input("Input the cost of the edge: "))
        mod = self._graphs[self._current_index_of_graph_list].change_cost(vertex_x, vertex_y, cost)
        if mod:
            print("Cost modified successfully!")
        else:
            print("Cannot modify the cost, the edge does not exist!")

    def get_in_degree_ui(self):
        vertex = int(input("Input the vertex:"))
        degree = self._graphs[self._current_index_of_graph_list].in_degree(vertex)
        if degree == -1:
            print("The vertex does not exist!")
        else:
            print("The in degree of the vertex {} is {}.".format(vertex, degree))

    def get_out_degree_ui(self):
        vertex = int(input("Input the vertex:"))
        degree = self._graphs[self._current_index_of_graph_list].out_degree(vertex)
        if degree == -1:
            print("The vertex does not exist!")
        else:
            print("The out degree of the vertex {} is {}.".format(vertex, degree))

    def check_if_edge_ui(self):
        vertex_x = int(input("Enter x = "))
        vertex_y = int(input("Enter y = "))
        edge = self._graphs[self._current_index_of_graph_list].find_if_edge(vertex_x, vertex_y)
        if edge is not False:
            print("({}, {}) is an edge and its cost is {}!".format(vertex_x, vertex_y, edge))
        else:
            print("({}, {}) is not an edge!".format(vertex_x, vertex_y))

    def copy_current_graph_ui(self):
        copy = self._graphs[self._current_index_of_graph_list].make_copy()
        self._graphs.append(copy)

    def print_menu(self):
        print("Graph MENU:\n"

              "1. Create a random graph with specified number of vertices and edges.\n"
              "2. Read the graph from a text file.\n"
              "3. Write the graph in a text file.\n"
              "4. Switch the graph.\n"
              "5. Get the number of vertices.\n"
              "6. Get the number of edges.\n"
              "7. Get the in degree of a vertex.\n"
              "8. Get the out degree of a vertex.\n"
              "9. Parse all the vertices.\n"
              "10. List the outbound edges of a given vertex.\n"
              "11. List all outbound vertices of the graph.\n"
              "12. List the inbound edges of a given vertex.\n"
              "13. List all inbound vertices of the graph. \n"
              "14. List the edges and their costs.\n"
              "15. Add a vertex.\n"
              "16. Remove a vertex.\n"
              "17. Add an edge.\n"
              "18. Remove an edge.\n"
              "19. Modify the cost of an edge.\n"
              "20. Check if there is an edge between two given vertices.\n"
              "21. Make a copy of the graph.\n"
              "22. Add an empty graph.\n"
              "23. Find the lowest length path from end_vertex to star_vertex using Backward BFS\n"
            "26. Check if the graph is a DAG and sort the activities topologically; if the graph is a DAG, find a highest cost path between 2 vertices.\n"

              "0. EXIT.\n")

    def lowest_len_path_bbfs_end_vertex_ui(self):
        start_vertex = int(input("Choose start vertex: "))
        print()
        end_vertex = int(input("Choose end vertex: "))
        print()
        parents = self._graphs[self._current_index_of_graph_list].shortest_path(
            self._graphs[self._current_index_of_graph_list], start_vertex, end_vertex)
        if len(parents) == 0:
            print("no such path found\n")
        else:
            print("Shortest path contains the following vertices from END -> BEGINNING:\n ")
            length = len(parents) - 1
            print("And its length is: ", length)
            print()
        for elem in parents:
            print(elem)

    # Lab 3 HW
    def floydWarshall(self):
        self.__floydWarshallMatrix, self.__floydWarshallPaths = self._graphs[self._current_index_of_graph_list].floydWarshall()

        for i in range(len(self._graphs[self._current_index_of_graph_list].parseKeys())):
            print(self.__floydWarshallMatrix[i])
        print()
        for i in range(len(self._graphs[self._current_index_of_graph_list].parseKeys())):
            print(self.__floydWarshallPaths[i])
        print("Floyd-Warshall matrix computed.")

    # Lab 3 HW
    def printLowestCostWalk(self):
        print("start vertex:")
        start = int(input())
        print("end vertex:")
        end = int(input())
        if start not in self._graphs[self._current_index_of_graph_list].parseKeys() or end not in self._graphs[self._current_index_of_graph_list].parseKeys():
            print("No such vertices")
            return
        if start == end:
            print("Cost from a vertex to itself is always 0")
        elif self.__floydWarshallMatrix[start][end] == math.inf:
            print("No possible walk from start to end.")
        else:
            """Print the cost"""
            print("Cost:", end='')
            print(self.__floydWarshallMatrix[start][end])

            print("Path:", end='')
            """Reconstruct the path"""

            path = [start]
            while start != end:
                start = self.__floydWarshallPaths[start][end]
                path.append(start)
            print(path)

    def topo_sort_ui(self):
        #sorted_list = self._graphs[self._current_index_of_graph_list].base_topo_sort()
        sorted_list = self._graphs[self._current_index_of_graph_list].toposort()
        if sorted_list == None:    #len(sorted_list)==0
        #if len(sorted_list)==0:
            print("The current graph is not a DAG!")
        else:
            print("The graph is a DAG and we have the following topological sorting: ")
            for vertex in sorted_list:
                print(str(vertex), end=' ')
            print("\n")
            vertex_begin = int(input("Enter source vertex = "))
            vertex_end = int(input("Enter end vertex = "))
            if vertex_begin not in self._graphs[self._current_index_of_graph_list].vertices or vertex_end not in self._graphs[self._current_index_of_graph_list].vertices:
                raise ValueError("One vertex/ both vertices entered might not exist.")
            else:
                distance, prev = self._graphs[self._current_index_of_graph_list].highest_cost_path(vertex_begin, vertex_end)
                if distance == float("-inf"):
                    raise ValueError("There is no walk from {} to {}".format(vertex_begin, vertex_end))
                path = []
                vertex = vertex_end
                path.append(vertex)
                while prev[vertex] != -1:
                    path.append(prev[vertex])
                    vertex = prev[vertex]
                print("The cost of the highest cost path is: {}".format(str(distance)))
                # print the path
            print("The path is: ", end=' ')
            for index in range(len(path) - 1, -1, -1):
                print(path[index], end=' ')
            print("\n")


    def start(self):
        print("Hello!")
        done = False
        self.add_empty_graph()
        print("The current graph is an empty graph!")
        command_dict = {"1": self.create_random_graph_ui, "2": self.read_graph_from_file_ui,
                        "3": self.write_graph_to_file_ui, "4": self.switch_graph_ui,
                        "5": self.get_number_of_vertices_ui, "6": self.get_number_of_edges_ui,
                        "7": self.get_in_degree_ui, "8": self.get_out_degree_ui,
                        "9": self.parse_all_vertices, "10": self.list_outbound,
                        "11": self.list_all_outbound, "12": self.list_inbound,
                        "13": self.list_all_inbound, "14": self.list_all_costs,
                        "15": self.add_vertex_ui, "16": self.delete_vertex_ui,
                        "17": self.add_edge_ui, "18": self.remove_edge_ui,
                        "19": self.modify_cost_ui, "20": self.check_if_edge_ui,
                        "21": self.copy_current_graph_ui, "22": self.add_empty_graph,
                        "23": self.lowest_len_path_bbfs_end_vertex_ui,
                        "24": self.floydWarshall,
                        "25": self.printLowestCostWalk,
                        "26": self.topo_sort_ui}
        while not done:
            try:
                self.print_menu()
                option = input("Choose an option from the Graph menu: \n")
                if option == "0":
                    done = True
                    print("Done!")
                elif option in command_dict:
                    command_dict[option]()
                else:
                    print("Bad input!\n")
            except ValueError as ve:
                print(str(ve))
            except FileNotFoundError as fnfe:
                print(str(fnfe).strip("[Errno 2] "))


UI().start()

# d
