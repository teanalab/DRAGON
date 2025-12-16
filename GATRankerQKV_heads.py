# import torch
# import torch.nn as nn
# from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv


class GATRanker(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_heads=16):
        super(GATRanker, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        # Define GCN layers
        # self.conv1 = GraphConv(in_feats, hidden_size)
        # self.conv2 = GraphConv(hidden_size, hidden_size)

        # Define two GAT layers instead of GCN layers.
        # For the first GAT layer, we set the output dimension per head to be hidden_size//num_heads.
        self.gat1 = GATConv(in_feats, hidden_size // num_heads, num_heads=num_heads)
        # After flattening, the feature dimension becomes hidden_size.
        # The second GAT layer then takes these features and outputs num_heads copies of a smaller vector,
        # so that after flattening we get back to hidden_size.
        self.gat2 = GATConv(hidden_size, hidden_size // num_heads, num_heads=num_heads)

        # Multi-headed attention layers
        self.q_w = nn.Linear(hidden_size, hidden_size)
        self.k_w = nn.Linear(hidden_size, hidden_size)
        self.v_w = nn.Linear(hidden_size, hidden_size)

        # Final output projection after concatenating multi-head outputs
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Batch normalization after attention
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU()
        )

        # Define final scoring layers
        ##TODO: I ADDED THEM
        # self.linear0001 = nn.Linear(hidden_size, 32)
        # self.leaky_relu0001 = nn.LeakyReLU()
        self.linear001 = nn.Linear(32, 16)
        self.leaky_relu001 = nn.LeakyReLU()
        self.linear01 = nn.Linear(16, 8)
        self.leaky_relu01 = nn.LeakyReLU()
        ##TODO: TILL HERE
        self.linear1 = nn.Linear(8, 4)
        self.leaky_relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(4, 1)
        self.last_layer = nn.Sigmoid()
        # self.bias = nn.Parameter(torch.zeros(1))
        # Initialize weights using Xavier initialization


    def forward(self, g, features) -> torch.Tensor:
        # Graph convolution layers
        # Apply first GAT layer:
        #   Output shape: [num_nodes, num_heads, hidden_size//num_heads]
        h = self.gat1(g, features)
        # Flatten the output: [num_nodes, hidden_size]
        h = h.flatten(1)
        # Apply the second GAT layer:
        #   Input: [num_nodes, hidden_size]
        #   Output: [num_nodes, num_heads, hidden_size//num_heads]
        h = self.gat2(g, h)
        # Flatten again to get [num_nodes, hidden_size]
        h = h.flatten(1)

        # Compute multi-headed attention
        # print(h.shape)
        num_nodes, _ = h.size()
        # batch_size = 1

        # Compute Q, K, V for all heads
        Q = self.q_w(h).view( num_nodes, self.num_heads, self.head_dim)
        K = self.k_w(h).view( num_nodes, self.num_heads, self.head_dim)
        V = self.v_w(h).view( num_nodes, self.num_heads, self.head_dim)

        # Transpose for attention computation: [batch_size, num_heads, num_nodes, head_dim]
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)

        # bias = self.bias
        # Scaled dot-product attention
        attention_scores = (Q @ K.transpose(-2, -1)) / (torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))

        ##TODO: I COMMENTED THIS LINE
        # attention_scores = (Q @ K.transpose(-2, -1)) / (torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) + 1e-9)

        # attention_scores = (Q @ K.transpose(-2, -1) + (bias * features) ) / (torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) + 1e-9)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to V
        attention_output = attention_weights @ V  # [batch_size, num_heads, num_nodes, head_dim]

        # # Concatenate all heads' outputs: reshape from [num_heads, num_nodes, head_dim] to [num_nodes, hidden_size]
        #first permutes the dimensions so that the nodes come first, then all attention head outputs,
        # and finally the head dimension. After reshaping (using .view(num_nodes, -1)),
        # the outputs of all heads are concatenated into one tensor of shape [num_nodes, hidden_size]
        # (because hidden_size = num_heads * head_dim).
        attention_output = attention_output.permute(1, 0, 2).contiguous().view( num_nodes, -1)

        # Project back to the hidden size
        attention_output = self.out_proj(attention_output)

        # Apply batch normalization
        # Reshape for BatchNorm1d: [batch_size * num_nodes, hidden_size]
        attention_output = attention_output.view(-1, attention_output.size(-1))
        attention_output = self.batch_norm(attention_output)

        # Reshape back to original dimensions: [batch_size, num_nodes, hidden_size]
        attention_output = attention_output.view(num_nodes, -1)

        # Pass through MLP
        attention_output = self.mlp(attention_output)

        # Final scoring
        ## TODO: I ADDED THEM
        # linear0001 = self.linear0001(attention_output)
        # leaky_relu0001 = self.leaky_relu0001(linear0001)
        linear001 = self.linear001(attention_output)
        leaky_relu001 = self.leaky_relu001(linear001)
        linear01 = self.linear01(leaky_relu001)
        leaky_relu01 = self.leaky_relu01(linear01)
        ##TODO: TILL HERE
        linear1 = self.linear1(leaky_relu01)
        leaky_relu1 = self.leaky_relu1(linear1)
        linear2 = self.linear2(leaky_relu1)
        score = self.last_layer(linear2)

        return score

