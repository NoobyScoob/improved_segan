import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, C, k=8, p=4):
        super(SelfAttention, self).__init__()
        self.L = 16384
        self.C = C
        self.k = C
        self.p = p

        # Define the weight matrices for the Q, K, V transformations
        self.Wq = nn.Conv1d(in_channels=C, out_channels=C//k, kernel_size=1)
        self.Wk = nn.Conv1d(in_channels=C, out_channels=C//k, kernel_size=1)
        self.Wv = nn.Conv1d(in_channels=C, out_channels=C//k, kernel_size=1)

        # Define the weight matrix for the output transformation
        self.Wo = nn.Conv1d(in_channels=C//k, out_channels=C, kernel_size=1)

        # Learnable parameter for the residual connection
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        batch_size = x.shape[0]

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        attention_scores = torch.matmul(Q.permute(0, 2, 1), K)
        attention_scores = F.softmax(attention_scores / (self.C//self.k)**0.5, dim=-1)

        O = torch.matmul(attention_scores, V.permute(0, 2, 1))
        O = O.permute(0, 2, 1)
        O = self.Wo(O)
        O = F.max_pool1d(O, kernel_size=1)
        F_hat = self.beta * O + x
        return F_hat