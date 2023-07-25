import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from nets.GraphMultiHeadAttention import GraphMultiHeadAttention
from nets.Encoder import GraphEncoder
from nets.Critic import Critic

class GraphAttentionModel(nn.Module):

    def __init__(self, num_customers, customer_feature, vehicle_feature, model_size=128, encoder_layer=3,
                 num_head=8, ff_size=128, tanh_xplor=10, edge_embedding_dim=128, greedy=False):
        super(GraphAttentionModel, self).__init__()

        # get models parameters for encoding-decoding
        self.model_size = model_size
        self.scaling_factor = self.model_size ** 0.5
        self.tanh_xplor = tanh_xplor
        self.greedy = greedy

        # Initialize encoder and embeddings
        self.customer_encoder = GraphEncoder(encoder_layer=encoder_layer,
                                             num_head=num_head,
                                             edge_dim_size=edge_embedding_dim,
                                             model_size=model_size,
                                             ff_size=ff_size)

        self.customer_embedding = nn.Linear(customer_feature, model_size)
        self.depot_embedding = nn.Linear(customer_feature, model_size)
        self.bn_nodes = nn.BatchNorm1d(model_size)
        self.bn_edges = nn.BatchNorm1d(edge_embedding_dim)
        self.edge_embedding = nn.Linear(1, edge_embedding_dim)
        self.fleet_attention = GraphMultiHeadAttention(num_head, vehicle_feature, model_size)
        self.vehicle_attention = GraphMultiHeadAttention(num_head, model_size)
        self.customer_projection = nn.Linear(self.model_size, self.model_size)  # TODO: MLP instaed of nn.Linear

        ## critic network
        self.critic = Critic(num_customers, ff_size)

    def encoder(self, env, customer_mask=None):
        customer_embed = torch.cat((self.depot_embedding(env.nodes[:, 0:1, :]),
                                    self.customer_embedding(env.nodes[:, 1:, :])), dim=1)
        if customer_mask is not None:
            customer_embed[customer_mask] = 0

        edge_embed = self.edge_embedding(env.edge_attributes)
        customer_embed = self.bn_nodes(customer_embed.permute(0, 2, 1)).permute(0, 2, 1)
        edge_embed = self.bn_edges(edge_embed.permute(0, 2, 1)).permute(0, 2, 1)

        self.customer_encoding = self.customer_encoder(customer_embed, edge_embed, mask=customer_mask)
        self.fleet_attention.precompute(self.customer_encoding)
        self.customer_representation = self.customer_projection(self.customer_encoding)
        if customer_mask is not None:
            self.customer_representation[customer_mask] = 0

    def decoder(self, env, old_action=None):

        fleet_representation = self.fleet_attention(env.vehicles,
                                                    mask=env.current_vehicle_mask)
        vehicle_query = fleet_representation.gather(1,
                                                    env.current_vehicle_index.unsqueeze(2).expand(-1, -1,
                                                                                                  self.model_size))

        vehicle_representation = self.vehicle_attention(vehicle_query,
                                                        fleet_representation,
                                                        fleet_representation)

        compact = torch.bmm(vehicle_representation,
                            self.customer_representation.transpose(2, 1))
        compact *= self.scaling_factor
        if self.tanh_xplor is not None:
            compact = self.tanh_xplor * compact.tanh()

        compact[env.current_vehicle_mask] = -float('inf')

        ###########################################################################################
        # waiting heuristic in case there is no customer, vehicle should wait at current location
        # if (current_vehicle[:, :, 5] != current_vehicle[:, :, 4]).all():
        #     compact.scatter_(2,
        #                      current_vehicle[:, :, 5].squeeze().long().unsqueeze(-1).unsqueeze(-1),
        #                      -self.tanh_xplor)
        # compact[:, :, 0] = -(self.tanh_xplor**1.1)
        # ##########################################################################################

        prop = F.softmax(compact, dim=-1)

        # step actions based on model act or evalaute
        if old_action is not None:
            dist = Categorical(prop)
            old_actions_logp = dist.log_prob(old_action[:, 1].unsqueeze(-1))
            is_done = float(env.done)

            entropy = dist.entropy()
            # entropy *= (1. - is_done)
            # old_actions_logp *= (1. - is_done)

            ## get values from critic networks
            val = self.critic(prop, env.current_vehicle_mask, old_action[:, 1].unsqueeze(-1))

            return old_action[:, 1].unsqueeze(-1), entropy, old_actions_logp, val

        else:
            dist = Categorical(prop)

            if self.greedy:
                _, customer_index = prop.max(dim=-1)
            else:
                customer_index = dist.sample()

            is_done = float(env.done)

            logp = dist.log_prob(customer_index)
            val = self.critic(prop, env.current_vehicle_mask, customer_index)
            # logp = logp*(1. - is_done)
            return customer_index, logp, val

    def evaluate(self, env, old_actions=None):
        env.reset()
        entropys, old_actions_logps, values = [], [], []
        steps = old_actions.size(0)

        for i in range(steps):
            if env.new_customer:
                self.encoder(env, env.customer_mask)

            old_action = old_actions[i, :, :]
            next_action = old_actions[i + 1, :, :] if i < steps - 1 else old_action
            next_vehicle_index = next_action[:, 0].unsqueeze(-1)
            customer_index, entropy, logp, val = self.decoder(env, old_action)
            env.step(customer_index, next_vehicle_index)

            old_actions_logps.append(logp.unsqueeze(1))
            entropys.append(entropy.unsqueeze(1))
            values.append(val)

        entropys = torch.cat(entropys, dim=1)
        num_e = entropys.ne(0).float().sum(1)
        entropy = entropys.sum(1) / num_e

        old_actions_logps = torch.cat(old_actions_logps, dim=1)
        old_actions_logps = old_actions_logps.sum(1)

        return entropy, old_actions_logps, values

    def act(self, env):

        env.reset()
        actions, logps, rewards, values = [], [], [], []
        while not env.done:
            if env.new_customer:
                self.encoder(env, env.customer_mask)
            customer_index, logp, val = self.decoder(env)
            actions.append((env.current_vehicle_index, customer_index))
            logps.append(logp.unsqueeze(1))
            rewards.append(env.step(customer_index))
            values.append(val)

        logps = torch.cat(logps, dim=1)
        logp_sum = logps.sum(dim=1)

        return actions, logp_sum, rewards, values

    def forward(self):
        raise NotImplementedError
