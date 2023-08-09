import torch


class DVRPSR_Environment:
    # vehicle coordinates(x_i,y_i), veh_time_time_budget, total_travel_time, last_customer, next(destination) customer
    vehicle_feature = 6
    # customers feature: coordinates (x_i,y_i), service time, arrival_time
    customer_feature = 4

    def __init__(self,
                 data=None,
                 nodes=None,
                 edges_attributes=None,
                 vehicle_count=2,
                 vehicle_speed=1,
                 vehicle_time_budget=1,
                 pending_cost=0.1,
                 dynamic_reward=1,
                 budget_penalty=2):

        self.vehicle_count = data.vehicle_count if data is not None else vehicle_count
        self.vehicle_speed = data.vehicle_speed if data is not None else vehicle_speed
        self.vehicle_time_budget = data.vehicle_time_budget if data is not None else vehicle_time_budget
        self.nodes = data.nodes if nodes is None else nodes
        self.edge_attributes = data.edges_attributes if edges_attributes is None else edges_attributes
        self.minibatch, self.nodes_count, _ = self.nodes.size()
        self.distance_matrix = self.edge_attributes.view((self.minibatch, self.nodes_count, self.nodes_count))
        self.pending_cost = pending_cost
        self.dynamic_reward = dynamic_reward
        self.budget_penalty = budget_penalty

    def _update_current_vehicles(self, dest, customer_index):
        # update vehicle previous and next customer id
        self.current_vehicle[:, :, 4] = self.current_vehicle[:, :, 5]
        self.current_vehicle[:, :, 5] = customer_index

        # get the distance from current vehicle to its next destination
        # Convert indices to integers
        current_idx = self.current_vehicle[:, :, 4].long()
        next_idx = self.current_vehicle[:, :, 5].long()
        dist = self.distance_matrix[torch.arange(self.minibatch).unsqueeze(1),
                                                 current_idx,
                                                 next_idx].view(self.minibatch, 1)
        # total travel time and
        # budget left while travelling to destination nodes
        budget = dist / self.vehicle_speed + dest[:, :, 2]

        # update vehicle features based on destination nodes
        self.current_vehicle[:, :, :2] = dest[:, :, :2]
        self.current_vehicle[:, :, 2] -= budget
        self.current_vehicle[:, :, 3] += budget

        # update vehicles states
        self.vehicles = self.vehicles.scatter(1,
                                              self.current_vehicle_index[:, :, None].expand(-1, -1, self.vehicle_feature),
                                              self.current_vehicle)
        return dist

    def _done(self, customer_index):
        self.vehicle_done.scatter_(1, self.current_vehicle_index,
                                      torch.logical_or((self.current_vehicle[:, :, 2] <= 0), (customer_index == 0)))

        self.done = torch.logical_or(self.vehicle_done.all(),
                                     (self.pending_customers == 0).all())

    def _update_mask(self, customer_index):
        self.new_customer = False
        self.served.scatter_(1, customer_index, customer_index > 0)
        self.pending_customers = (self.served ^ True).float().sum(-1, keepdim=True) - 1

        # cost for a vehicle to go to customer and back to deport considering service duration
        current_idx = self.current_vehicle[:, :, 5].long()
        dist_vehicle_customer_depot = self.distance_matrix[torch.arange(self.minibatch).unsqueeze(1), current_idx, :].squeeze(1) +\
                                      self.distance_matrix[:, :, 0]
        cost = dist_vehicle_customer_depot / self.vehicle_speed
        cost += self.nodes[:, :, 2]
        overtime_mask = self.current_vehicle[:, :, None, 2] - cost.unsqueeze(1)
        overtime = torch.zeros_like(self.mask).scatter_(1,
                                                        self.current_vehicle_index[:, :, None].
                                                        expand(-1, -1, self.nodes_count),overtime_mask < 0)
        self.mask = self.mask | self.served[:, None, :] | overtime | self.vehicle_done[:, :, None]
        # masking of depot is set to False
        self.mask[:, :, 0] = 0

    # updating current vehicle to find the next available vehicle
    def _update_next_vehicle(self, veh_index=None):
        if veh_index is None:
            avail = self.vehicles[:, :, 3].clone()
            avail[self.vehicle_done] = float('inf')
            self.current_vehicle_index = avail.argmin(1, keepdim=True)
        else:
            self.current_vehicle_index = veh_index

        self.current_vehicle = self.vehicles.gather(1, self.current_vehicle_index[:, :, None].expand(-1, -1, self.vehicle_feature))
        self.current_vehicle_mask = self.mask.gather(1, self.current_vehicle_index[:, :, None].expand(-1, -1, self.nodes_count))

    def _update_dynamic_customers(self, veh_index):
        time = self.current_vehicle[:, :, 3].clone()
        reveal_dyn_reqs = torch.logical_and((self.customer_mask), (self.nodes[:, :, 3] <= time))
        if reveal_dyn_reqs.any():
            self.new_customer = True
            self.customer_mask = self.customer_mask ^ reveal_dyn_reqs
            self.mask = self.mask ^ reveal_dyn_reqs[:, None, :].expand(-1, self.vehicle_count, -1)
            self.vehicle_done = torch.logical_and(self.vehicle_done, (reveal_dyn_reqs.any(1) ^ True).unsqueeze(1))

            # avail vehicle only when time budget is left
            time_violate = (self.vehicles[:, :, 2] <= 0)
            self.vehicle_done = torch.logical_or(self.vehicle_done, time_violate)

            self.vehicles[:, :, 3] = torch.max(self.vehicles[:, :, 3], time)
            self._update_next_vehicle(veh_index)

    def reset(self):
        # reset vehicle (minibatch*veh_count*veh_feature)
        self.vehicles = self.nodes.new_zeros((self.minibatch, self.vehicle_count, self.vehicle_feature))
        self.vehicles[:, :, :2] = self.nodes[:, :1, :2]
        self.vehicles[:, :, 2] = self.vehicle_time_budget

        # reset vehicle done
        self.vehicle_done = self.nodes.new_zeros((self.minibatch, self.vehicle_count), dtype=torch.bool)
        self.done = False

        # initialize reward as tour length
        self.tour_length = torch.zeros((self.minibatch, 1)).to(self.nodes.device)

        # reset cust_mask
        self.customer_mask = self.nodes[:, :, 3] > 0

        # reset new customers and served customer since now to zero (all false)
        self.new_customer = True
        self.served = torch.zeros_like(self.customer_mask)
        self.pending_customers = (self.served ^ True).float().sum(-1, keepdim=True) - 1

        # reset mask (minibatch*veh_count*nodes)
        self.mask = self.customer_mask[:, None, :].repeat(1, self.vehicle_count, 1)

        # reset current vehicle index, current vehicle, current vehicle mask
        self.current_vehicle_index = self.nodes.new_zeros((self.minibatch, 1), dtype=torch.int64)
        self.current_vehicle = self.vehicles.gather(1,
                                                    self.current_vehicle_index[:, :, None].
                                                    expand(-1, -1, self.vehicle_feature))
        self.current_vehicle_mask = self.mask.gather(1,
                                                     self.current_vehicle_index[:, :, None].
                                                     expand(-1, -1, self.nodes_count))
    #
    # def get_reward(self):
    #     if self.done:
    #         return self.tour_length + self.pending_cost*self.pending_customers

    def step(self, customer_index, veh_index=None):
        dest = self.nodes.gather(1, customer_index[:, :, None].expand(-1, -1, self.customer_feature))
        dist = self._update_current_vehicles(dest, customer_index)
        self._done(customer_index)
        self._update_mask(customer_index)
        self._update_next_vehicle(veh_index)
        self._update_dynamic_customers(veh_index)

        self.tour_length += dist

        reward = +dist

        if self.done:
            served_customers = torch.sum(self.served.float(), dim=1)
            pending_customers = torch.sum((self.served ^ True) & (self.nodes[:, :, 3] >= 0), dim=1) - 1

            # Define hyperparameters for reward shaping
            alpha = 0.75  # Weight for distance minimization
            beta = 50.0  # Weight for customer satisfaction (served/pending ratio)

            # Calculate tour length reward
            tour_length_reward = self.tour_length

            # Calculate customer satisfaction reward
            unserved_ratio = pending_customers / (served_customers + pending_customers + 1e-6)
            satisfaction_reward = beta * unserved_ratio

            # Combine the rewards using a weighted sum
            reward += alpha * tour_length_reward + (1 - alpha) * satisfaction_reward.unsqueeze(-1)

            return reward
        else:
            return reward

    # def step(self, customer_index, veh_index=None):
    #     dest = self.nodes.gather(1, customer_index[:, :, None].expand(-1, -1, self.customer_feature))
    #     dist = self._update_current_vehicles(dest, customer_index)
    #     self._done(customer_index)
    #     self._update_mask(customer_index)
    #     self._update_next_vehicle(veh_index)
    #     self._update_dynamic_customers(veh_index)
    #
    #     self.tour_length += dist
    #
    #     reward = +dist
    #
    #     if self.done:
    #         # penalty for all and static pending customers
    #         pending_customers = torch.logical_and((self.served ^ True),
    #                                               (self.nodes[:, :, 3] >= 0)).float().sum(-1, keepdim=True) - 1
    #         reward += self.pending_cost * pending_customers
    #         return reward
    #     else:
    #         return reward


    def state_dict(self, dest_dict=None):
        if dest_dict is None:
            dest_dict = {'vehicles': self.vehicles,
                         'vehicle_done': self.vehicle_done,
                         'served': self.served,
                         'mask': self.mask,
                         'current_vehicle_index': self.current_vehicle_index}
        else:
            dest_dict["vehicles"].copy_(self.vehicles)
            dest_dict["vehicle_done"].copy_(self.vehicle_done)
            dest_dict["served"].copy_(self.served)
            dest_dict["mask"].copy_(self.mask)
            dest_dict["current_vehicle_index"].copy_(self.current_vehicle_index)

        return dest_dict

    def load_state_dict(self, state_dict):
        self.vehicles.copy_(state_dict["vehicles"])
        self.vehicle_done.copy_(state_dict["vehicle_done"])
        self.served.copy_(state_dict["served"])
        self.mask.copy_(state_dict["mask"])
        self.current_vehicle_index.copy_(state_dict["current_vehicle_index"])
        self.current_vehicle = self.vehicles.gather(1,
                                                    self.current_vehicle_index[:, :, None].
                                                    expand(-1, -1, self.vehicle_feature))
        self.current_vehicle_mask = self.mask.gather(1, self.current_vehicle_index[:, :, None].
                                                     expand(-1, -1, self.customer_feature))