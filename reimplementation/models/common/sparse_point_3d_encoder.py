import torch
import torch.nn as nn
from ..utils.model_utils import linear_relu_ln

class SparsePoint3DEncoder(nn.Module):
    def __init__(
        self, 
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
    ):
        super(SparsePoint3DEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.input_dims = num_sample * coords_dim
        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(self.input_dims)

    def forward(self, anchor: torch.Tensor):
        pos_feat = self.pos_fc(anchor)  
        return pos_feat