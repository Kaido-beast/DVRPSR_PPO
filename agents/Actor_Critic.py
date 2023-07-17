import torch
import torch.nn as nn
from nets import GraphAttentionModel
from nets import GraphAttentionModel_v1
from agents.Critic import Critic


class Actor_Critic(nn.Module):

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
                 edge_embedding_dim=128,
                 greedy=False):
        super(Actor_Critic, self).__init__()
        model = GraphAttentionModel(customer_feature, vehicle_feature, model_size, encoder_layer,
                                    num_head, ff_size_actor, tanh_xplor, edge_embedding_dim, greedy)
        self.actor = model

        self.critic = Critic(model, customers_count, ff_size_critic)

    def act(self, env, old_actions=None, is_update=False):
        actions, logps, rewards = self.actor(env)
        return actions, logps, rewards

    def evaluate(self, env, old_actions, is_update):
        #print('critic values **************************')
        values = self.critic(env)

        #print('agent evaluate ******************************')
        entropys, old_logps, _ = self.actor(env, old_actions, is_update)
        return entropys, old_logps, values
