import os
import time

import torch
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import numpy as np
from torch.utils.data import Dataset, DataLoader

from TrainPPOAgent import TrainPPOAgent
from problems import DVRPSR_Environment
from problems import DVRPSR_Dataset

def train():
    class RunBuilder():
        @staticmethod
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    params = OrderedDict(customer_feature=[4],
                         vehicle_feature=[8],
                         customers_count=[21],
                         model_size=[128],
                         encoder_layer=[3],
                         num_head=[8],
                         ff_size_actor=[128],
                         ff_size_critic=[512],
                         tanh_xplor=[10],
                         edge_embedding_dim=[64],
                         greedy=[False],
                         learning_rate=[3e-4],
                         ppo_epoch=[3],
                         batch_size=[256],
                         entropy_value=[0.2],
                         epsilon_clip=[0.2],
                         epoch=[40],
                         timestep=[2])

    runs = RunBuilder.get_runs(params)

    for customer_feature, vehicle_feature, customers_count, model_size, \
            encoder_layer, num_head, ff_size_actor, ff_size_critic, tanh_xplor, edge_embedding_dim, greedy, \
            learning_rate, ppo_epoch, batch_size, entropy_value, epsilon_clip, epoch, timestep in runs:
        data = torch.load("./data/train/DVRPSR_{}_{}_{}_{}/normalized_street.pth".format(0.025, 0.5, 2, 400 ))
        env = DVRPSR_Environment

        trainppo = TrainPPOAgent(customer_feature, vehicle_feature, customers_count, model_size,
                                 encoder_layer, num_head, ff_size_actor, ff_size_critic, tanh_xplor,
                                 edge_embedding_dim, greedy, learning_rate, ppo_epoch, batch_size,
                                 entropy_value, epsilon_clip, epoch, timestep)



        trainppo.run_train(data, env, batch_size)


train()