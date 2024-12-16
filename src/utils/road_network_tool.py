import os
import osmnx as ox
import networkx as nx
import math


def load_graph(file_path):
    """
    this function is only used by function "download_map_and_convert_to_graph()"
    @param file_path:
    @return:
    """

    print("begin load road network")
    loaded_graph = ox.load_graphml(file_path)
    print("the road network has been loaded")
    print(loaded_graph)
    return loaded_graph


def download_map_and_convert_to_graph(bbox, file_path, network_type='drive', simplify=True):
    """
    this function is used to download maps from OpenStreetmap based on latitude and longitude ranges and convert it
    to MultiDiGraph based road network. if the road network had been downloaded, it will directly load the
    downloaded road network from location.
    where the file was saved
    @param bbox: [north bounder, south bounder, east bounder, west bounder]
    @param file_path: the file path for saving/loading the road network
    @param network_type: the type of the road network: drive, all
    @param simplify:
    @return: a MultiDiGraph object that stores road network information
    """
    file_name = str(bbox[0]) + "_" + str(bbox[1]) + "_ " + str(bbox[2]) + "_" + str(
        bbox[3]) + "_" + network_type + "_map_data.graphml"
    if os.path.exists(file_path + file_name):
        return load_graph(file_path + file_name)
    try:
        # Download street network data from OpenStreetMap
        print("begin download")
        G = ox.graph_from_bbox(bbox=bbox, network_type=network_type, simplify=simplify)

        # 检查图是否成功下载
        if G is None or len(G.nodes) == 0:
            raise ValueError("图对象为空或没有节点，请检查下载参数。")
        if nx.is_directed(G):
            G = G.to_undirected()
        # 移除孤立的节点
        G.remove_nodes_from(list(nx.isolates(G)))

        # 移除孤立的边
        edges_to_remove = [edge for edge in G.edges() if not G.has_edge(*edge)]
        G.remove_edges_from(edges_to_remove)


        ox.save_graphml(G, filepath=file_path + file_name)
        print("download finished")
        print(G)
        return G
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class RoadNetworkTool:
    def __init__(self, file_path, bbox, network_type, grid_number=16):
        """

        @param file_path: the file path for saving/loading the road network
        @param bbox: [north bounder, south bounder, east bounder, west bounder]
        @param network_type: the type of the road network: drive, all
        @param grid_number:
        """
        self.bbox = bbox
        self.min_lat = min(self.bbox[0], self.bbox[1])
        self.min_long = min(self.bbox[2], self.bbox[3])
        self.file_path = file_path
        self.graph = download_map_and_convert_to_graph(bbox, file_path, network_type)
        self.grid_size = abs(bbox[0] - bbox[1]) / grid_number
        self.num_lat = 0
        self.num_long = 0
        self.grid_index = self.grid_index_construction()

    def grid_index_construction(self):
        """
        construct the grid index
        mainly for acceleration of function "find_nearest_node"
        @return: a list: each item of this list represents a grid and it stores all the nodes located in this grid
        """
        grid_index_list = []
        self.num_lat = num_lat = math.ceil(abs(self.bbox[0] - self.bbox[1]) / self.grid_size)
        self.num_long = num_long = math.ceil(abs(self.bbox[2] - self.bbox[3]) / self.grid_size)
        for i in range(num_lat * num_long):
            grid_index_list.append([])
        for node in self.graph.nodes(data=True):
            """
            node 
            (811012413, {'y': 29.8077023, 'x': 104.7077632, 'ref': '16', 'highway': 'motorway_junction', 'street_count': 3})
            """
            latitude = node[1]["y"]
            longitude = node[1]["x"]
            lat_num = math.floor((latitude - self.min_lat) / self.grid_size)
            long_num = math.floor((longitude - self.min_long) / self.grid_size)
            index = lat_num * num_lat + long_num
            grid_index_list[index].append(node)
        return grid_index_list

    def find_nearest_node_(self, coord):
        """
        find the nearest node without the acceleration of grid index
        @param coord: (latitude, longitude)
        @return: the ID of the nearest node
        """
        min_distance = float('inf')
        nearest_node = None
        for node in self.graph.nodes(data=True):
            node_coord = (node[1]['y'], node[1]['x'])
            distance = ((coord[0] - node_coord[0]) ** 2 + (coord[1] - node_coord[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_node = node[0]
        return nearest_node

    def find_nearest_node(self, coord):
        """
        find the nearest node with the acceleration of grid index
        @param coord: (latitude, longitude)
        @return: the ID of the nearest node
        """
        min_distance = float('inf')
        nearest_node = None
        candidate_node_list = []
        lat_num = math.floor((coord[0] - self.min_lat) / self.grid_size)
        long_num = math.floor((coord[1] - self.min_long) / self.grid_size)
        index_list = []
        if lat_num - 1 >= 0 and long_num - 1 >= 0:
            index_list.append((lat_num - 1) * self.num_lat + (long_num - 1))
        if lat_num - 1 >= 0:
            index_list.append((lat_num - 1) * self.num_lat + long_num)
        if lat_num - 1 >= 0 and long_num + 1 < self.num_long:
            index_list.append((lat_num - 1) * self.num_lat + (long_num + 1))
        if long_num - 1 >= 0:
            index_list.append(lat_num * self.num_lat + (long_num - 1))
        if long_num - 1 >= 0 and lat_num + 1 < self.num_lat:
            index_list.append((lat_num + 1) * self.num_lat + (long_num - 1))
        index_list.append(lat_num * self.num_lat + long_num)
        if long_num + 1 < self.num_long:
            index_list.append(lat_num * self.num_lat + (long_num + 1))
        if lat_num + 1 < self.num_lat:
            index_list.append((lat_num + 1) * self.num_lat + long_num)
        if lat_num + 1 < self.num_lat and long_num + 1 < self.num_long:
            index_list.append((lat_num + 1) * self.num_lat + (long_num + 1))
        if lat_num - 2 >= 0 and long_num - 1 >= 0:
            index_list.append((lat_num - 2) * self.num_lat + (long_num - 1))
        if lat_num - 2 >= 0:
            index_list.append((lat_num - 2) * self.num_lat + long_num)
        if lat_num - 2 >= 0 and long_num + 1 < self.num_long:
            index_list.append((lat_num - 2) * self.num_lat + (long_num + 1))
        if lat_num - 1 >= 0 and long_num + 2 < self.num_long:
            index_list.append((lat_num - 1) * self.num_lat + (long_num + 2))
        if long_num + 2 < self.num_long:
            index_list.append(lat_num * self.num_lat + (long_num + 2))
        if lat_num + 1 < self.num_lat and long_num + 2 < self.num_long:
            index_list.append((lat_num + 1) * self.num_lat + (long_num + 2))
        if long_num - 1 >= 0 and lat_num + 2 < self.num_lat:
            index_list.append((lat_num + 2) * self.num_lat + (long_num - 1))
        if lat_num + 2 < self.num_lat:
            index_list.append((lat_num + 2) * self.num_lat + long_num)
        if lat_num + 2 < self.num_lat and long_num + 1 < self.num_long:
            index_list.append((lat_num + 2) * self.num_lat + (long_num + 1))
        if lat_num - 1 >= 0 and long_num - 2 >= 0:
            index_list.append((lat_num - 1) * self.num_lat + (long_num - 2))
        if long_num - 2 >= 0:
            index_list.append(lat_num * self.num_lat + (long_num - 2))
        if long_num - 2 >= 0 and lat_num + 1 < self.num_lat:
            index_list.append((lat_num + 1) * self.num_lat + (long_num - 2))
        for index in index_list:
            if 0 <= index < len(self.grid_index):
                candidate_node_list.extend(self.grid_index[index])

        for node in candidate_node_list:
            node_coord = (node[1]['y'], node[1]['x'])
            distance = ((coord[0] - node_coord[0]) ** 2 + (coord[1] - node_coord[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_node = node[0]
        if nearest_node is None:
            nearest_node = self.find_nearest_node_(coord)
        return nearest_node

    def find_the_shortest_path(self, origin_point, destination_point, method="dijkstra"):
        """
        given two points with their latitudes and longitudes, it finds the shortest path between these two points
        this function is implemented using ”networkx“
        @param method: the default algorithm for shortest path finding is "dijkstra"
        @param origin_point: (latitude, longitude)
        @param destination_point: (latitude, longitude)
        @return: (the length of the shortest path (unit: meter), the route of the shortest path, which is a list and
        each item of which is the ID of corresponding node)
        """
        origin_node = self.find_nearest_node(origin_point)
        destination_node = self.find_nearest_node(destination_point)
        return nx.shortest_path_length(self.graph, origin_node, destination_node, weight='length', method=method), \
            nx.shortest_path(self.graph, origin_node, destination_node, weight='length', method=method)

# file_path = "D:\\数据集\\"
# # north, south, east, west = 52.4786135, 52.332759, 9.6903624, 9.8481922
# # 成都市
# north, south, east, west = 20.1, 19.6, 111, 110.1
# bbox = [north, south, east, west]
# road_network = RoadNetworkTool(file_path, bbox, "all", 6)

# if __name__ == '__main__':
#     file_path = "D:\\数据集\\"
#     bbox = [31.4872339, 29.7753633, 104.8716203, 103.1195788]
#     road_network = RoadNetworkTool(file_path, bbox, "drive", 6)
#
#     # 1478082936, 1478084270, 104.123161, 30.679787, 104.15508, 30.62972, 4.41
#     point1 = (30.679787, 104.123161)
#     point2 = (30.62972, 104.15508)
#     shortest_path_length, shortest_path = road_network.find_the_shortest_path(point1, point2)
#     print(shortest_path_length, shortest_path)
