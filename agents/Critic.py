import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Critic(nn.Module):

    # critic will take environment as imput and ouput the values for loss function
    # which is basically the estimation of complexity of actions

    def __init__(self, model, customers_count, ff_size=512):
        super(Critic, self).__init__()

        self.model = model
        self.ff_layer1 = nn.Linear(customers_count, ff_size)
        self.ff_layer2 = nn.Linear(ff_size, customers_count)

    def eval_step(self, env, compatibility, customer_index):
        compact = compatibility.clone()
        compact[env.current_vehicle_mask] = 0

        value = self.ff_layer1(compact)
        value = F.relu(value)
        value = self.ff_layer2(value)

        val = value.gather(2, customer_index.unsqueeze(1)).expand(-1, 1, -1)
        return val.squeeze(1)

    def __call__(self, env):
        self.model.encode_customers(env)
        env.reset()

        values = []

        while not env.done:
            _vehicle_presentation = self.model.vehicle_representation(env.vehicles,
                                                                      env.current_vehicle_index,
                                                                      env.current_vehicle_mask)
            compatibility = self.model.score_customers(_vehicle_presentation)
            prop = self.model.get_prop(compatibility, env.current_vehicle_mask)
            dist = Categorical(prop)
            customer_index = dist.sample()

            values.append(self.eval_step(env, compatibility, customer_index))
            env.step(customer_index)

        return torch.cat(values, dim=1).sum(dim=1)


