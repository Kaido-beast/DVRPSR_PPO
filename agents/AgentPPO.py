import torch
import torch.nn as nn
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

        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size
        self.entropy_value = entropy_value
        self.epsilon_clip = epsilon_clip
        # initialize the Adam optimizer
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.MSE_loss = nn.MSELoss(reduction='none')
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
        assert std != 0. and not torch.isnan(std), 'Need nonzero std'
        norm_advantage = (advantage - advantage.mean()) / (std + 1e-8)
        return norm_advantage

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
        lr_scheduler = LambdaLR(self.optim, lr_lambda= lambda f: 0.96**epoch)
        env = env if env is not None else DVRPSR_Environment

        for i in range(self.ppo_epoch):
            self.policy.train()
            loss_t, norm_R, critic_R, loss_a, loss_mse, loss_e, ratios = [], [], [], [], [], [], []
            for batch_index, minibatch_data in enumerate(data_loader):
                minibatch_data.to(device)
                nodes = minibatch_data.nodes
                edge_attributes = minibatch_data.edge_attributes
                nodes = nodes.view(self.batch_size, self.customers_count, self.customer_feature)
                edge_attributes = edge_attributes.view(self.batch_size, self.customers_count * self.customers_count, 1)
                old_actions_for_env = minibatch_data.actions.view(self.batch_size, max_length, 2).permute(1, 0, 2)
                dyna_env = env(None, nodes, edge_attributes, *env_params)
                entropy, log_probs, values = self.policy.evaluate(dyna_env, old_actions_for_env, True)

                R_norm = minibatch_data.rewards.to(device)
                R_norm = self.advantage_normalization(R_norm)
                mse_loss = self.MSE_loss(values, R_norm)

                ratio = torch.exp(log_probs - minibatch_data.log_probs).squeeze(1)

                # PPO advantage
                advantage = (R_norm - values.detach())
                # PPO overall loss function
                actor_loss1 = ratio * advantage
                actor_loss2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage
                actor_loss = torch.min(actor_loss1, actor_loss2)

                # total loss
                loss = actor_loss + 0.5 * mse_loss - self.entropy_value*entropy.squeeze(1)

                # optimizer and backpropogation
                self.optim.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()
                lr_scheduler.step()

                norm_R.append(torch.mean(R_norm.detach()).item())
                loss_t.append(torch.mean(loss.detach()).item())
                loss_a.append(torch.mean(actor_loss.detach()).item())
                loss_mse.append(torch.mean(mse_loss.detach()).item())
                loss_e.append(torch.mean(entropy.detach()).item())
                critic_R.append(torch.mean(values.detach()).item())
                ratios.append(torch.mean(ratio.detach()).item())

        self.old_policy.load_state_dict(self.policy.state_dict())

        return loss_t, loss_a, loss_mse, loss_e, norm_R, critic_R, ratios


if __name__ == '__main__':
    raise Exception('Cannot be called from main')