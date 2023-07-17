import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import LambdaLR
import time
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profiler

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
                 ff_size_critic=128,
                 tanh_xplor=10,
                 edge_embedding_dim=128,
                 greedy=False,
                 learning_rate=3e-4,
                 ppo_epoch=3,
                 batch_size=128,
                 entropy_value=0.2,
                 epsilon_clip=0.2,
                 max_grad_norm=2):

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
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
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

    def advantage_normalization(self, advantage):

        std = advantage.std()

        #assert std != 0. and not torch.isnan(std), 'Need nonzero std'
        if std==0:
            std = 1

        norm_advantage = (advantage - advantage.mean()) / (std + 1e-8)
        return norm_advantage

    def pad_actions(self, actions):
        max_len = max(a.size(0) for a in actions)
        padded_actions = torch.zeros(len(actions), max_len, actions[0].size(1), dtype=actions[0].dtype)
        for i, a in enumerate(actions):
            length = a.size(0)
            padded_actions[i, :length] = a
        return padded_actions

    def update(self, memory, epoch, data=None, env=None, env_params=None, device=None):

        old_nodes = torch.stack(memory.nodes)
        old_edge_attributes = torch.stack(memory.edge_attributes)
        old_rewards = torch.stack(memory.rewards).unsqueeze(-1)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(-1)
        padded_actions = torch.stack(memory.actions)
        max_length = padded_actions.size(1)

        # create update data for PPO
        datas = []
        for i in range(old_nodes.size(0)):
            data_to_load = Data(nodes=old_nodes[i],
                                edge_attributes=old_edge_attributes[i],
                                actions=padded_actions[i],
                                rewards=old_rewards[i],
                                log_probs=old_log_probs[i])

            datas.append(data_to_load)

        self.policy.to(device)


        data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=False)
        lr_scheduler = LambdaLR(self.optim, lr_lambda= lambda f:0.96**epoch)

        self.entropy_value = self.entropy_value * (0.99**epoch)
        self.epsilon_clip = self.epsilon_clip * (0.99**epoch)
        env = env if env is not None else DVRPSR_Environment

        times, losses, rewards, critic_rewards = [], [], [], []

        for i in range(self.ppo_epoch):

            epoch_start = time.time()
            start = epoch_start
            self.policy.train()

            loss_t, norm_R, critic_R, loss_a, loss_mse, loss_e = [], [], [], [], [], []

            for batch_index, minibatch_data in enumerate(data_loader):

                self.batch_index += 1
                # minibatch_data =  minibatch_data.to(device)


                nodes = minibatch_data.nodes.to(device)
                customer_mask = None
                edge_attributes = minibatch_data.edge_attributes.to(device)
                nodes = nodes.view(self.batch_size, self.customers_count, self.customer_feature)
                edge_attributes = edge_attributes.view(self.batch_size, self.customers_count * self.customers_count, 1)

                old_actions_for_env = minibatch_data.actions.view(self.batch_size, max_length, 2).permute(1,0,2).to(
                    device)
                #print(nodes.size(), edge_attributes.size(), old_actions_for_env.size())
                dyna_env = env(None, nodes, edge_attributes, *env_params)
                entropy, log_probs, values = self.policy.evaluate(dyna_env, old_actions_for_env, True)

                # Wrap the forward pass with the profiler
                # with profiler.profile(use_cuda=True) as prof:
                #     self.policy.evaluate(dyna_env, old_actions_for_env, is_update=False)
                #
                # # Print profiling results
                # print(prof.key_averages().table(sort_by="cpu_time_total"))

                # normalize the rewards and get the MSE loss with critics values

                R_norm = minibatch_data.rewards.to(device).squeeze(-1)
                #print(R_norm.size)

                R_norm = self.advantage_normalization(R_norm)

                mse_loss = self.MSE_loss(R_norm, values)

                # PPO ration (r(0)_t)
                ratio = torch.exp(log_probs - minibatch_data.log_probs.to(device))

                # PPO advantage
                advantage = R_norm - values.detach()

                #print(R.size(), R_norm.size(), values.size())

                # PPO overall loss function
                actor_loss1 = ratio * advantage
                actor_loss2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage

                actor_loss = torch.min(actor_loss1, actor_loss2)

                # print('ratio {}, advantage {}, actor_loss1 {}, actor_loss2 {}'.format(ratio,
                #                                                                    advantage,
                #                                                                    actor_loss1,
                #                                                                    actor_loss2))

                # total loss
                loss = actor_loss +  mse_loss - self.entropy_value * entropy

                # optimizer and backpropogation
                # self.optimizer.zero_grad()
                self.optim.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # self.optimizer.step()
                self.optim.step()

                lr_scheduler.step()

                # print('Rnorm {}, old_rewards {}, critic_values {}, actor_loss {}, entropy {} total_loss {}'.format(R_norm,
                #                                                                                                 R,
                #                                                                                                 values,
                #                                                                                                 actor_loss,
                #                                                                                                 entropy,
                #                                                                                                 loss))

                norm_R.append(torch.mean(R_norm.detach()).item())

                loss_t.append(torch.mean(loss.detach()).item())
                loss_a.append(torch.mean(actor_loss.detach()).item())
                loss_mse.append(torch.mean(mse_loss.detach()).item())
                loss_e.append(torch.mean(entropy.detach()).item())

                critic_R.append(torch.mean(values.detach()).item())

        self.old_policy.load_state_dict(self.policy.state_dict())

        #print(' after update rewards {}, losses{}, critic_rewards {}'.format(rewards, losses, critic_rewards))

        return loss_t, loss_a, loss_mse, loss_e, norm_R, critic_R
