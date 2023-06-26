import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphMultiHeadAttention(nn.Module):

    def __init__(self, num_head, query_size, key_size=None, value_size=None, edge_dim_size=None, bias=False):

        super(GraphMultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.query_size = query_size

        self.key_size = self.query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size
        self.edge_dim_size = self.query_size // 2 if edge_dim_size is None else edge_dim_size

        self.scaling_factor = self.key_size ** -0.5

        self.keys_per_head = self.key_size // self.num_head
        self.values_per_head = self.value_size // self.num_head
        self.edge_size_per_head = self.edge_dim_size

        self.edge_embedding = nn.Linear(self.edge_dim_size, self.edge_size_per_head, bias=bias)
        self.query_embedding = nn.Linear(self.query_size, self.num_head * self.keys_per_head, bias=bias)
        self.key_embedding = nn.Linear(self.key_size, self.num_head * self.keys_per_head, bias=bias)
        self.value_embedding = nn.Linear(self.value_size, self.num_head * self.values_per_head, bias=bias)
        self.recombine = nn.Linear(self.num_head * self.values_per_head, self.value_size, bias=bias)

        self.K_project_pre = None
        self.V_project_pre = None

        self.initialize_weights()

    def initialize_weights(self):
        # TODO: add xavier initialziation as well

        nn.init.uniform_(self.query_embedding.weight, -self.scaling_factor, self.scaling_factor)
        nn.init.uniform_(self.key_embedding.weight, -self.scaling_factor, self.scaling_factor)
        inv_sq_dv = self.value_size ** -0.5
        nn.init.uniform_(self.value_embedding.weight, -inv_sq_dv, inv_sq_dv)

    def precompute(self, keys, values=None):

        values = keys if values is None else values

        size_KV = keys.size(-2)

        self.K_project_pre = self.key_embedding(keys).view(
            -1, size_KV, self.num_head, self.keys_per_head).permute(0, 2, 3, 1)

        self.V_project_pre = self.value_embedding(values).view(
            -1, size_KV, self.num_head, self.values_per_head).permute(0, 2, 1, 3)

    def forward(self, queries, keys=None, values=None, edge_attributes=None, mask=None, edge_mask=None):

        *batch_size, size_Q, _ = queries.size()

        # get queries projection
        Q_project = self.query_embedding(queries).view(
            -1, size_Q, self.num_head, self.keys_per_head).permute(0, 2, 1, 3)

        # get keys projection
        if keys is None:
            if self.K_project_pre is None:
                size_KV = size_Q
                K_project = self.key_embedding(queries).view(
                    -1, size_KV, self.num_head, self.keys_per_head).permute(0, 2, 3, 1)
            else:
                size_KV = self.K_project_pre.size(-1)
                K_project = self.K_project_pre
        else:
            size_KV = keys.size(-2)
            K_project = self.key_embedding(keys).view(
                -1, size_KV, self.num_head, self.keys_per_head).permute(0, 2, 3, 1)

        # get values projection
        if values is None:
            if self.V_project_pre is None:
                V_project = self.value_embedding(queries).view(
                    -1, size_KV, self.num_head, self.values_per_head).permute(0, 2, 1, 3)
            else:
                V_project = self.V_project_pre
        else:
            V_project = self.value_embedding(values).view(
                -1, size_KV, self.num_head, self.values_per_head).permute(0, 2, 1, 3)

        # calculate the compability
        attention = Q_project.matmul(K_project)
        attention *= self.scaling_factor

        # if edge attributes are required
        if edge_attributes is not None:
            # TODO: edge mask (is it required)
            edge_project = self.edge_embedding(edge_attributes).view(
                -1, size_Q, size_Q, self.edge_size_per_head)

            # get enhanced attention inclusing edge attributes
            attention_expanded = attention.unsqueeze(-1).expand(-1, -1, -1, -1, self.edge_size_per_head)

            # Expand edge attributes to match the number of attention heads
            edge_project_expanded = edge_project.unsqueeze(1).expand(-1, attention.size(1), -1, -1, -1)
            #print(attention_expanded.size(), edge_project_expanded.size())

            attention = attention_expanded * edge_project_expanded
            #print(attention.size())
            attention = attention.mean(-1)

            # print(attention.size())

        if mask is not None:

            if mask.numel() * self.num_head == attention.numel():
                m = mask.view(-1, 1, size_Q, size_KV).expand_as(attention)
            else:
                m = mask.view(-1, 1, 1, size_KV).expand_as(attention)

            attention[m.bool()] = -float('inf')

        attention = F.softmax(attention, dim=-1)
        attention = attention.matmul(V_project).permute(0, 2, 1, 3).contiguous().view(
            *batch_size, size_Q, self.num_head * self.values_per_head)

        output = self.recombine(attention)

        return output

