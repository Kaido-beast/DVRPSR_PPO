import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from nets import GraphMultiHeadAttention
from nets.Encoder import GraphEncoder


class GraphAttentionModel(nn.Module):

    def __init__(self, customer_feature, vehicle_feature, model_size=128, encoder_layer=3,
                 num_head=8, ff_size=128, tanh_xplor=10, edge_embedding_dim=64, greedy=False):
        super().__init__()

        # get models parameters for encoding-decoding
        self.model_size = model_size
        self.scaling_factor = self.model_size ** 0.5
        self.tanh_xplor = tanh_xplor
        self.greedy = greedy

        # Initialize encoder and embeddings
        self.customer_encoder = GraphEncoder(encoder_layer=3, num_head=8, model_size=model_size, ff_size=ff_size)
        self.customer_embedding = nn.Linear(customer_feature, model_size)
        self.depot_embedding = nn.Linear(customer_feature, model_size)

        # initialize edge embedding
        self.edge_embedding = nn.Linear(1, edge_embedding_dim)

        # Initialize vehicle embedding and encoding
        # self.vehicle_embedding = nn.Linear(vehicle_feature, ff_size, bias=False)

        self.fleet_attention = GraphMultiHeadAttention(num_head, vehicle_feature, model_size)

        self.vehicle_attention = GraphMultiHeadAttention(num_head, model_size)

        # customer projection
        self.customer_projection = nn.Linear(self.model_size, self.model_size)  # TODO: MLP instaed of nn.Linear

    def encode_customers(self, env, customer_mask=None):

        customer_emb = torch.cat((self.depot_embedding(env.nodes[:, :1, :]),
                                  self.customer_embedding(env.nodes[:, 1:, :])), dim=1)
        if customer_mask is not None:
            customer_emb[customer_mask] = 0

        edge_emb = self.edge_embedding(env.edge_attributes)

        self.customer_encoding = self.customer_encoder(customer_emb, edge_emb, mask=customer_mask)

        self.fleet_attention.precompute(self.customer_encoding)

        self.customer_representation = self.customer_projection(self.customer_encoding)
        if customer_mask is not None:
            self.customer_representation[customer_mask] = 0

    def vehicle_representation(self, vehicles, vehicle_index, vehicle_mask=None):

        fleet_representation = self.fleet_attention(vehicles, mask=vehicle_mask)
        vehicle_query = fleet_representation.gather(0, vehicle_index.unsqueeze(2).expand(-1, -1, self.model_size))

        return self.vehicle_attention(vehicle_query, fleet_representation, fleet_representation)

    def score_customers(self, vehicle_representation):

        # print(vehicle_representation.size(), self.customer_representation.size())
        compact = torch.bmm(vehicle_representation,
                            self.customer_representation.transpose(2, 1))
        compact *= self.scaling_factor

        if self.tanh_xplor is not None:
            compact = self.tanh_xplor * compact.tanh()

        return compact

    def get_prop(self, compact, vehicle_mask=None):
        compact[vehicle_mask] = -float('inf')
        compact = F.softmax(compact, dim=-1)
        return compact

    def step(self, env, old_action=None):

        _vehicle_representation = self.vehicle_representation(env.vehicles,
                                                              env.current_vehicle_index,
                                                              env.current_vehicle_mask)

        compact = self.score_customers(_vehicle_representation)
        prop = self.get_prop(compact, env.current_vehicle_mask)
        #print(compact.size())

        # step actions based on model act or evalaute
        if old_action is not None:

            # get entropy
            dist = Categorical(prop)
            old_actions_logp = dist.log_prob(old_action[:, 1].unsqueeze(-1))
            entropy = dist.entropy()

            is_done = float(env.done)

            entropy *= (1. - is_done)
            old_actions_logp *= (1. - is_done)
            return old_action[:, 1].unsqueeze(-1), entropy, old_actions_logp


        else:
            dist = Categorical(prop)

            if self.greedy:
                _, customer_index = p.max(dim=-1)
            else:
                customer_index = dist.sample()

            is_done = float(env.done)

            logp = dist.log_prob(customer_index)
            logp *= (1. - is_done)

            return customer_index, logp

    def forward(self, env, old_actions=None, is_update=False):

        if is_update:
            env.reset()
            entropys, old_actions_logps = [], []

            steps = old_actions.size(0)

            for i in range(steps):
                if env.new_customer:
                    self.encode_customers(env, env.customer_mask)

                old_action = old_actions[i, :, :]
                next_action = old_actions[i + 1, :, :] if i < steps - 1 else old_action

                next_vehicle_index = next_action[:, 0].unsqueeze(-1)
                # print(next_vehicle_index)

                customer_index, entropy, logp = self.step(env, old_action)

                env.step(customer_index, next_vehicle_index)

                old_actions_logps.append(logp)
                entropys.append(entropy)

            entropys = torch.cat(entropys, dim=1)
            num_e = entropys.ne(0).float().sum(1)
            entropy = entropys.sum(1) / num_e

            old_actions_logps = torch.cat(old_actions_logps, dim=1)
            old_actions_logps = old_actions_logps.sum(1)

            return entropy, old_actions_logps, 0

        else:
            env.reset()
            actions, logps, rewards = [], [], []

            while not env.done:
                if env.new_customer:
                    self.encode_customers(env, env.customer_mask)

                customer_index, logp = self.step(env)
                actions.append((env.current_vehicle_index, customer_index))
                logps.append(logp)
                rewards.append(env.step(customer_index))

            # actions = torch.cat(actions, dim=1)
            logps = torch.cat(logps, dim=1)
            logp = logps.sum(dim=1)

            rewards = torch.cat(rewards, dim=1)
            rewards = rewards.sum(dim=1)

            return actions, logp, rewards



