import torch
import random
import math
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import networkx as nx
import sys

class DVRPSR_Dataset_euclidean(Dataset):
    customer_feature = 4  # customer features location (x_i,y_i) and duration of service(d), appearance (u)

    @classmethod
    def create_data(cls,
                    batch_size=2,
                    vehicle_count=2,
                    vehicle_speed=20,  # km/hr
                    Lambda=0.025,  # request rate per min
                    dod=0.5,
                    horizon=400,
                    fDmean=10,
                    fDstd=2.5):

        # static customer counts V = Lambda*horizon*(1-dod)/(dod+0.5)
        V_static = int(Lambda * horizon * (1 - dod) / (dod) + 0.5)

        # total customer count
        V = int(Lambda * horizon / (dod) + 0.5)

        size = (batch_size, V, 1)

        # initialize the graph of vienna network
        graph = cls.initialize_graph()

        # get the coordinates of customers
        data_vienna = pd.read_csv('../vienna_data/vienna_cordinates.csv')

        # get depot coordinates: Id, xcoords, ycoords
        depot = cls.get_depot_location(data_vienna)

        # get location of customers: id, xcoords, ycoords
        locations = cls.get_customers_coordinates(data_vienna, batch_size, V, depot)

        # get edges index and attributes, which is distance between one node to others n_i*n_j
        edges_index, edges_attributes = cls.get_edges_attributes(batch_size, graph, depot, locations, V)

        ### generate Static_Dynamic customer requests
        dynamic_request = cls.generateRandomDynamicRequests(batch_size,
                                                            V,
                                                            V_static,
                                                            fDmean,
                                                            fDstd,
                                                            Lambda,
                                                            horizon)

        customers = torch.zeros((batch_size, V, cls.customer_feature))
        customers[:, :, :2] = locations[:, :, 1:]
        customers[:, :, 2:4] = dynamic_request

        depo = torch.zeros((batch_size, 1, cls.customer_feature))
        depo[:, :, 0:2] = torch.from_numpy(depot[0][1:])
        depo[:, :, 2] = 0

        nodes = torch.cat((depo, customers), 1)

        dataset = cls(vehicle_count, vehicle_speed, horizon, nodes, V,
                      edges_index, edges_attributes, customer_mask=None)

        return dataset

    def __init__(self, vehicle_count, vehicle_speed, horizon, nodes, V,
                 edges_index, edges_attributes, customer_mask=None):

        self.vehicle_count = vehicle_count
        self.vehicle_speed = vehicle_speed
        self.nodes = nodes
        self.vehicle_time_budget = horizon
        self.edges_index = edges_index
        self.edges_attributes = edges_attributes

        self.batch_size, self.nodes_count, d = self.nodes.size()

        if d != self.customer_feature:
            raise ValueError("Expected {} customer features per nodes, got {}".format(
                self.customer_feature, d))

        self.customer_mask = customer_mask
        self.customer_count = V

    def initialize_graph():

        coordinates = pd.read_csv("../vienna_data/vienna_dist.csv", header=None, sep=' ')
        coordinates.columns = ['coord1', 'coord2', 'dist']
        graph = nx.DiGraph()

        # add the rows to the graph for shortest path and distance calculations
        for _, row in coordinates.iterrows():
            graph.add_edge(row['coord1'], row['coord2'], weight=row['dist'])

        return graph

    def precompute_shortest_path(graph, start_node, end_node):

        shortest_path = nx.shortest_path(graph, start_node, end_node)

        # TODO: distance need to be normalized afterwords
        shortest_path_length = sum(graph.get_edge_data(u, v)['weight']
                                   for u, v in zip(shortest_path, shortest_path[1:]))

        return shortest_path, shortest_path_length

    def get_distanceLL(lat1, lon1, lat2, lon2):

        R = 6371  # Radius of the Earth in kilometers

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def get_NearestNodeLL(lat, lon, lats, lons):
        nearest = (-1, sys.float_info.max)
        for i in range(len(lats)):
            dist = DVRPSR_Dataset_euclidean.get_distanceLL(lat, lon, lats[i], lons[i])
            if dist < nearest[1]:
                nearest = (i, dist)
        return nearest[0]

    def get_depot_location(data_vienna):

        ll = (48.178808, 16.438460)
        lat = ll[0] / 180 * math.pi
        lon = ll[1] / 180 * math.pi
        lats = data_vienna['lats']
        lons = data_vienna['lons']
        depot = DVRPSR_Dataset_euclidean.get_NearestNodeLL(lat, lon, lats, lons)
        depot_coordinates = np.array(data_vienna[data_vienna['id'] == depot][['id', 'xcoords', 'ycoords']])

        return depot_coordinates

    def get_customers_coordinates(data_vienna, batch_size, customers_count, depot):

        torch.manual_seed(42)

        # Excluding depot id from the customers selection
        data_vienna_without_depot = data_vienna[data_vienna['id'] != int(depot[0][0])].reset_index()

        # Sample customers indices for all batches at once
        sampled_customers = torch.multinomial(torch.tensor(data_vienna_without_depot['id'], dtype=torch.float32),
                                              num_samples=batch_size * customers_count, replacement=True)

        sampled_customers = sampled_customers.reshape(batch_size, customers_count)

        # Gather the sampled locations using the indices
        sampled_locations = data_vienna_without_depot.loc[sampled_customers.flatten()].reset_index(drop=True)

        # Reshape the locations to match the batch size
        locations = sampled_locations.groupby(sampled_locations.index // customers_count)

        # Create PyTorch tensors for the batched data
        locations_tensors = []
        for _, batch in locations:
            id_tensor = torch.tensor(batch['id'].values, dtype=torch.long)
            coords_tensor = torch.tensor(batch[['xcoords', 'ycoords']].values, dtype=torch.float32)
            batch_tensor = torch.cat((id_tensor.unsqueeze(1), coords_tensor), dim=1)
            locations_tensors.append(batch_tensor)

        return torch.stack(locations_tensors)

    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5

    def get_edges_attributes(batch_size, graph, depot, locations, V):

        # all customers ID inclusing depot

        print('Initialzing edges')
        edge_depot = torch.zeros((batch_size, 1, 2))
        edge_depot[:, :, 0] = depot[0][1]
        edge_depot[:, :, 1] = depot[0][2]
        edge_data = torch.cat((edge_depot, locations[:, :, 1:3]), dim=1)

        # generate edge index
        edges_index = []

        for i in range(V + 1):
            for j in range(V + 1):
                edges_index.append([i, j])
        edges_index = torch.LongTensor(edges_index)
        edges_index = edges_index.transpose(dim0=0, dim1=1)

        # generate nodes attributes
        edges_batch = []

        for batch in edge_data:
            edges = torch.zeros((V + 1, V + 1, 1), dtype=torch.float32)
            for i, node1 in enumerate(batch):
                for j, node2 in enumerate(batch):
                    distance = DVRPSR_Dataset_euclidean.c_dist(node1, node2)
                    edges[i][j][0] = distance

            edges = edges.reshape(-1, 1)
            edges_batch.append(edges)

        return edges_index, torch.stack(edges_batch)

    def generateRandomDynamicRequests(batch_size=2,
                                      V=20,
                                      V_static=10,
                                      fDmean=10,
                                      fDstd=2.5,
                                      Lambda=0.025,
                                      horizon=400,
                                      dep=0,
                                      u=0):
        gen = random.Random()
        gen.seed()  # uses the default system seed
        unifDist = gen.random  # uniform distribution
        durDist = lambda: max(0.01, gen.gauss(fDmean, fDstd))  # normal distribution with fDmean and fDstd

        # TODO: in actual data , we need to add a depo node with corrdinate, which should be removed from selected
        #       nodes as well.

        requests = []
        for b in range(batch_size):

            static_request = []
            dynamic_request = []
            u = 0

            while True:
                unif = unifDist()
                u += -(1 / Lambda) * math.log(unif)
                if u > horizon or len(dynamic_request) > (V - V_static + 2):
                    break
                d = round(durDist(), 2)
                while d <= 0:
                    d = round(durDist(), 2)

                dynamic_request.append([d, round(u, 2)])

            for j in range(V - len(dynamic_request)):
                d = round(durDist(), 2)
                while d <= 0:
                    d = round(durDist(), 2)
                static_request.append([d, 0])

            request = static_request + dynamic_request
            random.shuffle(request)
            requests.append(request)

        return torch.tensor(requests)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        if self.customer_mask is None:
            return self.nodes[i], self.edges_attributes[i]
        else:
            return self.nodes[i], self.customer_mask[i], self.edges_attributes[i]

    def nodes_generate(self):
        if self.customer_mask is None:
            yield from self.nodes
        else:
            yield from (n[m ^ 1] for n, m in zip(self.nodes, self.customer_mask))

    def normalize(self):
        loc_max, loc_min = self.nodes[:, :, :2].max().item(), self.nodes[:, :, :2].min().item()
        loc_max -= loc_min
        edge_max_length = self.edges_attributes.max().item()

        self.nodes[:, :, :2] -= loc_min
        self.nodes[:, :, :2] /= loc_max
        self.nodes[:, :, 2:] /= self.vehicle_time_budget

        self.vehicle_speed *= self.vehicle_time_budget / edge_max_length
        self.vehicle_time_budget = 1
        self.edges_attributes /= edge_max_length
        return loc_max, 1

    def save(self, folder_path):
        torch.save({
            'veh_count': self.vehicle_count,
            'veh_speed': self.vehicle_speed,
            'nodes': self.nodes,
            'edges_index': self.edges_index,
            'edges_attributes': self.edges_attributes,
            'customer_count': self.customer_count,
            'customer_mask': self.customer_mask
        }, folder_path)

    @classmethod
    def load(cls, folder_path):
        return cls(**torch.load(folder_path))

data = DVRPSR_Dataset_euclidean.create_data(2,2)