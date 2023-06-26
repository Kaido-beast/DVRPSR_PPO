import torch
import os.path
from itertools import repeat, zip_longest

def formate_old_actions(actions):
    old_actions = []

    for action in actions:
        old_action = []
        for i in range(action[0].size(0)):
            old_action.append([action[0][i].item(), action[1][i].item()])
        old_actions.append(old_action)
    return old_actions


def _pad_with_zeros(src_it):
    yield from src_it
    yield from repeat(0)

def eval_apriori_routes(dyna, routes, rollout_count):
    mean_cost = dyna.nodes.new_zeros(dyna.minibatch)
    for c in range(rollout_count):
        dyna.reset()
        routes_it = [[_pad_with_zeros(route) for route in inst_routes] for inst_routes in routes]
        rewards = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[n][i.item()])]
                for n,i in enumerate(dyna.current_vehicle_index)], dtype = torch.int64)
            rewards.append( dyna.step(cust_idx) )
        mean_cost += torch.stack(rewards).sum(dim = 0).squeeze(-1)
    return mean_cost / rollout_count