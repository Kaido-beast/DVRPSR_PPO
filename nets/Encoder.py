import torch
import torch.nn as nn
import torch.nn.functional as F
from nets import GraphMultiHeadAttention

class GraphEncoderlayer(nn.Module):

    def __init__(self, num_head, model_size, ff_size, edge_dim_size):
        super().__init__()

        self.attention = GraphMultiHeadAttention(num_head, query_size=model_size, edge_dim_size=edge_dim_size)
        self.BN1 = nn.BatchNorm1d(model_size)
        self.FFN_layer1 = nn.Linear(model_size, ff_size)

        self.FFN_layer2 = nn.Linear(ff_size, model_size)
        self.BN2 = nn.BatchNorm1d(model_size)

    def forward(self, h, e=None, mask=None):
        h_attn = self.attention(h, edge_attributes=e, mask=mask)
        h_bn = self.BN1((h_attn + h).permute(0, 2, 1)).permute(0, 2, 1)

        h_layer1 = F.relu(self.FFN_layer1(h_bn))
        h_layer2 = self.FFN_layer2(h_layer1)

        h_out = self.BN2((h_bn + h_layer2).permute(0, 2, 1)).permute(0, 2, 1)

        if mask is not None:
            h_out[mask] = 0

        return h_out


class GraphEncoder(nn.Module):

    def __init__(self, encoder_layer, num_head, model_size, ff_size, edge_dim_size):
        super().__init__()

        for l in range(encoder_layer):
            self.add_module(str(l), GraphEncoderlayer(num_head, model_size, ff_size, edge_dim_size))

    def forward(self, h_in, e_in=None, mask=None):

        h = h_in
        e = e_in

        for child in self.children():
            h = child(h, e , mask=mask)
        return h
