import torch.nn as nn
from nets import GraphAttentionModel


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
        model = GraphAttentionModel(customers_count, customer_feature, vehicle_feature, model_size, encoder_layer,
                                       num_head, ff_size_actor, tanh_xplor, edge_embedding_dim, greedy)
        self.actor = model

    def act(self, env, old_actions=None, is_update=False):
        actions, logps, rewards, state_values = self.actor.act(env)
        return actions, logps, rewards, state_values

    def evaluate(self, env, old_actions, is_update):
        entropys, old_logps, values = self.actor.evaluate(env, old_actions)
        return entropys, old_logps, values
