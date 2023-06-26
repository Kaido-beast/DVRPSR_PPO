import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import LambdaLR
import time

from agents.Actor_Critic import Actor_Critic
from problems import DVRPSR_Environment


class AgentPPO:

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
                 max_grad_norm = 2):

        self.policy = Actor_Critic(customer_feature, vehicle_feature, customers_count, model_size,
                                   encoder_layer, num_head, ff_size_actor, ff_size_critic,
                                   tanh_xplor, edge_embedding_dim, greedy)

        self.old_policy = Actor_Critic(customer_feature, vehicle_feature, customers_count, model_size,
                                       encoder_layer, num_head, ff_size_actor, ff_size_critic,
                                       tanh_xplor, edge_embedding_dim, greedy)

        self.old_policy.load_state_dict(self.policy.state_dict())

        # ppo update parameters
        # self.learning_rate = learning_rate
        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size
        self.entropy_value = entropy_value
        self.epsilon_clip = epsilon_clip
        self.batch_index = 1

        # initialize the Adam optimizer
        #self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.MSE_loss = nn.MSELoss()

        # actor-critic parameters
        self.customer_feature = customer_feature
        self.vehicle_feature = vehicle_feature
        self.customers_count = customers_count
        self.model_size = model_size
        self.encoder_layer = encoder_layer
        self.num_head = num_head
        self.ff_size_actor = ff_size_actor
        self.ff_size_critic = ff_size_critic
        self.tanh_xplor = tanh_xplor
        self.edge_embedding_dim = edge_embedding_dim
        self.greedy = greedy
        self.max_grad_norm = max_grad_norm

        self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

    def advantage_normalization(self, advantage):

        std = advantage.std()

        assert std != 0. and not torch.isnan(std), 'Need nonzero std'

        norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return norm_advantage

    def pad_actions(self, actions, device):
        max_len = max([a.size(0) for a in actions])
        padded_actions = []
        for a in actions:
            pad_length = max_len - a.size(0)
            padded_a = F.pad(a, (0, 0, 0, pad_length), value=0).to(device)
            padded_actions.append(padded_a)
        return torch.stack(padded_actions), max_len

    def update(self, memory, epoch, data=None, env=None, optim=None, lr_scheduler=None, device=None):

        old_nodes = torch.stack(memory.nodes)
        old_edge_attributes = torch.stack(memory.edge_attributes)
        old_rewards = torch.stack(memory.rewards).unsqueeze(-1)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(-1)

        # preprocessing on old actions
        padded_actions, max_length = self.pad_actions(memory.actions, device)
        #print(old_nodes.size(), old_edge_attributes.size(), old_rewards.size(), old_log_probs.size(),padded_actions.size())

        # create update data for PPO
        datas = []

        # print(memory.actions.size())

        for i in range(old_nodes.size(0)):
            data_to_load = Data(nodes=old_nodes[i],
                                edge_attributes=old_edge_attributes[i],
                                actions=padded_actions[i],
                                rewards=old_rewards[i],
                                log_probs=old_log_probs[i])

            datas.append(data_to_load)
        # print(datas[0], self.batch_size)

        self.policy.to(device)

        data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=False)

        # scheduler = LambdaLR(self.optimizer, lr_lambda=lambda f: 0.96 ** epoch)
        value_buffer = 0

        env = env if env is not None else DVRPSR_Environment

        for i in range(self.ppo_epoch):

            self.policy.train()
            epoch_start = time.time()
            start = epoch_start

            self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

            for batch_index, minibatch_data in enumerate(data_loader):

                self.batch_index += 1
                minibatch_data =  minibatch_data.to(device)

                if data.customer_mask is None:
                    nodes = minibatch_data.nodes
                    customer_mask = None
                    edge_attributes = minibatch_data.edge_attributes

                nodes = nodes.view(self.batch_size, self.customers_count, self.customer_feature)
                edge_attributes = edge_attributes.view(self.batch_size, self.customers_count * self.customers_count, 1)

                old_actions_for_env = minibatch_data.actions.view(self.batch_size, max_length, 2).permute(1, 0, 2)
                #print(old_actions_for_env)

                dyna_env = env(data, nodes, customer_mask, edge_attributes)

                entropy, log_probs, values = self.policy.evaluate(dyna_env, old_actions_for_env, True)

                # normalize the rewards and get the MSE loss with critics values
                R = minibatch_data.rewards
                R_norm = self.advantage_normalization(R)

                mse_loss = self.MSE_loss(R_norm, values.squeeze(-1))

                # PPO ration (r(0)_t)
                ratio = torch.exp(log_probs - minibatch_data.log_probs)

                # PPO advantage
                advantage = R_norm - values.detach()

                # PPO overall loss function
                actor_loss1 = ratio * advantage
                actor_loss2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage

                actor_loss = torch.min(actor_loss1, actor_loss2)

                # total loss
                loss = actor_loss + 0.5 * mse_loss - self.entropy_value * entropy

                # optimizer and backpropogation
                #self.optimizer.zero_grad()
                optim.zero_grad()
                loss.mean().backward(retain_graph = True)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #self.optimizer.step()
                optim.step()

                lr_scheduler.step()

                self.rewards.append(torch.mean(R_norm.detach()).item())
                self.losses.append(torch.mean(loss.detach()).item())
                self.critic_rewards.append(torch.mean(values.detach()).item())

        self.old_policy.load_state_dict(self.policy.state_dict())

        return self.rewards, self.losses, self.critic_rewards

if __name__ == '__main__':
    raise Exception('Cannot be called from main')
