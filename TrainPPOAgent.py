import os
import time
import torch
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from collections import namedtuple
from itertools import product
import numpy as np
from agents import AgentPPO
from utils import Memory
from utils.Misc import formate_old_actions
import tqdm
from tqdm import tqdm

class TrainPPOAgent:

    def __init__(self,
                 customer_feature,
                 vehicle_feature,
                 customers_count,
                 model_size=128,
                 encoder_layer=3,
                 num_head=8,
                 ff_size_actor=128,
                 ff_size_critic=512,
                 tanh_xplor=10,
                 edge_embedding_dim=64,
                 greedy=False,
                 learning_rate=3e-4,
                 ppo_epoch=3,
                 batch_size=256,
                 entropy_value=0.2,
                 epsilon_clip=0.2,
                 epoch=40,
                 timestep=2,
                 max_grad_norm=2):

        self.greedy = greedy
        self.memory = Memory()
        self.batch_size = batch_size
        self.customers_count = customers_count
        self.update_timestep = timestep
        self.epoch = epoch
        self.agent = AgentPPO(customer_feature, vehicle_feature, customers_count, model_size,
                              encoder_layer, num_head, ff_size_actor, ff_size_critic,
                              tanh_xplor, edge_embedding_dim, greedy, learning_rate,
                              ppo_epoch, batch_size, entropy_value, epsilon_clip, max_grad_norm)

    def run_train(self, args, datas, env, env_params, optim, lr_scheduler, device, epoch):

        train_data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=True)
        #print(self.batch_size)

        memory = Memory()
        self.agent.old_policy.to(device)

        epoch_loss = 0
        epoch_prop = 0
        epoch_val = 0
        epoch_c_val = 0

        self.agent.old_policy.train()
        times, losses, rewards1, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start_time = epoch_start

        with tqdm(train_data_loader, desc="Epoch #{: >3d}/{: <3d}".format(epoch + 1, args.epoch_count)) as progress:

            for batch_index, minibatch in enumerate(progress):

                if datas.customer_mask is None:
                    nodes, customer_mask, edge_attributes = minibatch[0].to(device), None, minibatch[1].to(device)

                nodes = nodes.view(self.batch_size, self.customers_count, 4)
                edge_attributes = edge_attributes.view(self.batch_size, self.customers_count * self.customers_count, 1)

                dyna_env = env(datas, nodes, customer_mask, edge_attributes, *env_params)

                actions, logps, rewards = self.agent.old_policy.act(dyna_env)

                ## formate the actions for memory
                actions = formate_old_actions(actions)
                actions = torch.tensor(actions)
                actions = actions.permute(1, 0, 2)

                actions = actions.to(device)
                logps = logps.to(device)
                rewards = rewards.to(device)

                #print(actions.size())

                memory.nodes.extend(nodes)
                memory.edge_attributes.extend(edge_attributes)
                memory.rewards.extend(rewards)
                memory.log_probs.extend(logps)
                memory.actions.extend(actions)

                if (batch_index + 1) % self.update_timestep == 0:
                    u_rewards, u_losses, u_critic_rewards = self.agent.update(memory, epoch, datas, env, optim, lr_scheduler, device)
                    #print(u_losses, u_critic_rewards)
                    memory.clear()

                prob = torch.stack([logps]).sum(dim=0).exp().mean()
                val = torch.stack([rewards]).sum(dim=0).mean()
                c_val = torch.tensor(u_critic_rewards).mean()
                u_losses = torch.tensor(u_losses).mean()

                progress.set_postfix_str("l={:.4g} p={:9.4g} val={:6.4g} c_val={:6.4g}".format(
                    u_losses.item(), prob.item(), val.item(), c_val.item()))

                epoch_loss += u_losses.item()
                epoch_prop += prob.item()
                epoch_val += val.item()
                epoch_c_val += c_val.item()

            return tuple(stats / args.iter_count for stats in (epoch_loss, epoch_prop, epoch_val, epoch_c_val))

    def test_epoch(self, args, env, agent, ref_costs):
        agent.eval()
        costs = env.nodes.new_zeros(env.minibatch)

        for _ in range(100):
            _, _, rewards = agent.act(env)
            costs += torch.stack([rewards]).sum(dim=0).squeeze(-1)

        costs = costs / 100

        mean = costs.mean()
        std = costs.std()
        gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

        print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
        return mean.item(), std.item(), gap.item()
