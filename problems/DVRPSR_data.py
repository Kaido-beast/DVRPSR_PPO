import torch
from torch.utils.data import Dataset
import os
from time import time
from problems import *
from problems.utils_edges_euclidean import *
# import problems.utils_data as utils
# import problems.utils_edges_euclidean as get_edges_euclidean
from utils.config import *

class DVRPSR_Dataset(Dataset):
    customer_feature = 4  # customer features location (x_i,y_i) and duration of service(d), appearance (u)

    @classmethod
    def create_data(cls,
                    batch_size=2,
                    vehicle_count=2,
                    vehicle_speed=1,  # 20 km/hr * normalized_factor
                    Lambda=0.025,  # request rate per min
                    dod=0.5,
                    horizon=400,
                    fDmean=10,
                    fDstd=2.5,
                    cust_location_range=(0, 100),
                    enclidean = True):

        # static customer counts V = Lambda*horizon*(1-dod)/(dod+0.5)
        V_static = int(Lambda * horizon * (1 - dod) / (dod) + 0.5)

        # total customer count
        V = int(Lambda * horizon / (dod) + 0.5)

        size = (batch_size, V, 1)

        # customer location selected uniformaly x_i,y_i ~ U(0,100)
        locations = torch.randint(*cust_location_range, (batch_size, V + 1, 2), dtype=torch.float)

        # get edges index and attributes, which is distance between one node to others n_i*n_j
        # get the real street data or euclidean data between coordinates (which is faster to compute)
        edges_index, edges_attributes = get_edges_attributes_parallel(batch_size, locations, V)

        ### generate Static_Dynamic customer requests
        dynamic_request = utils_data.generateRandomDynamicRequests(batch_size,
                                                            V,
                                                            V_static,
                                                            fDmean,
                                                            fDstd,
                                                            Lambda,
                                                            horizon)

        customers = torch.zeros((batch_size, V, cls.customer_feature))
        customers[:, :, :2] = locations[:, 1:]
        customers[:, :, 2:4] = dynamic_request

        depo = torch.zeros((batch_size, 1, cls.customer_feature))
        depo[:, :, 0:2] = locations[:, 0:1]
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
        #edge_max_length = self.edges_attributes[:, :, 0].max().item()

        self.nodes[:, :, :2] -= loc_min
        self.nodes[:, :, :2] /= loc_max
        self.nodes[:, :, 2:] /= self.vehicle_time_budget

        self.vehicle_speed *= self.vehicle_time_budget / loc_max
        self.vehicle_time_budget = 1
        self.edges_attributes /= loc_max
        return loc_max, 1

    def save(self, folder_path):
        torch.save({
            'veh_count': self.vehicle_count,
            'veh_speed': self.vehicle_speed,
            'nodes': self.nodes,
            'edges_index': self.edges_index,
            'edges_attributes': self.edges_attributes,
            'customer_count': self.customer_count,
            'cust_mask': self.cust_mask
        }, folder_path)

    @classmethod
    def load(cls, folder_path):
        return cls(**torch.load(folder_path))


if __name__ == '__main__':
    args = ParseArguments()
    start_time = time()

    train_test_val = 'validation'

    if train_test_val == 'train':
        batch_size = args.batch_size * args.iter_count
    elif train_test_val == 'test':
        batch_size = args.test_batch_size
    else:
        batch_size = 1000

    print(batch_size)
    vehicle_count = args.vehicle_count
    vehicle_speed = args.vehicle_speed
    Lambda = args.Lambda
    dod = args.dod
    horizon = args.horizon
    data = DVRPSR_Dataset.create_data(batch_size=batch_size,
                                      vehicle_count=vehicle_count,
                                      vehicle_speed=vehicle_speed,
                                      Lambda=Lambda,
                                      dod=dod,
                                      horizon=horizon)
    if train_test_val == 'train':
        data.normalize()

    end_time = time()

    # save the data
    folder_path = "../data/{}/{}_{}_{}_{}".format(train_test_val, Lambda, dod, vehicle_count, horizon)
    os.makedirs(folder_path, exist_ok=True)
    if train_test_val == 'train':
        torch.save(data, os.path.join(folder_path, "train_50.pth"))
    elif train_test_val == 'test':
        torch.save(data, os.path.join(folder_path, "test.pth"))
    else:
        torch.save(data, os.path.join(folder_path, "val_accuracy.pth"))

    print(f'Time to run {batch_size} batches is {end_time-start_time}')
    print(data.nodes[0], data.nodes[0].size())

