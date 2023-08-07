
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, customers_count, ff_size=128):
        super(Critic, self).__init__()

        self.ff_layer1 = nn.Linear(customers_count, ff_size)
        self.ff_layer2 = nn.Linear(ff_size, customers_count)  # Output is a single value

        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.ff_layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.ff_layer2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, model_compact, current_vehicle_mask=None, customer_index=None):
        compact = model_compact.clone()
        compact[current_vehicle_mask] = 0
        value = F.relu(self.ff_layer1(compact))
        value = self.ff_layer2(value)
        #val = value.gather(2, customer_index.unsqueeze(1)).squeeze(1)
        return value.squeeze(1)
