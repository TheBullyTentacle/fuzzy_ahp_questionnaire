import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List
from collections.abc import Iterable


# 1. Enhanced type hints for Python 3.12
class TGNMemory(torch.nn.Module):
    def __init__(self, node_dim: int, edge_feat_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_feat_dim = edge_feat_dim
        self.gru = torch.nn.GRUCell(node_dim * 2 + 1 + edge_feat_dim, node_dim)
        self.memories: dict[int, torch.nn.Parameter] = {}

    def init_memory(self, node: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.zeros(self.node_dim), requires_grad=False)

    def update_memory(self, node: int, message: torch.Tensor) -> None:
        if node not in self.memories:
            self.memories[node] = self.init_memory(node)
        self.memories[node] = self.gru(message, self.memories[node])


class GraphAttention(torch.nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(node_dim, num_heads=4, batch_first=True)

    def forward(self,
                src_emb: torch.Tensor,
                neighbor_embs: torch.Tensor,
                edge_feats: torch.Tensor) -> torch.Tensor:
        # Combine edge features with embeddings
        attn_input = torch.cat([src_emb.unsqueeze(1), neighbor_embs + edge_feats], dim=1)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        return attn_output[:, 0]


class PfoTGNRec(torch.nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 node_dim: int = 64,
                 edge_feat_dim: int = 1,
                 gamma: float = 1.0,
                 lambda_mv: float = 0.5):
        super().__init__()
        self.memory = TGNMemory(node_dim, edge_feat_dim)
        self.attention = GraphAttention(node_dim)
        self.gamma = gamma
        self.lambda_mv = lambda_mv

        # Vectorized embeddings
        self.user_emb = torch.nn.Embedding(num_users, node_dim)
        self.item_emb = torch.nn.Embedding(num_items, node_dim)

        # Portfolio tracking using tensor operations
        self.register_buffer('portfolio', torch.zeros((num_users, num_items), dtype=torch.bool))

    def compute_mv_scores(self,
                          candidates: torch.Tensor,
                          portfolio_items: torch.Tensor,
                          returns: torch.Tensor,
                          cov_matrix: torch.Tensor) -> torch.Tensor:
        """Vectorized MV score calculation"""
        if portfolio_items.numel() == 0:
            return returns[candidates] / self.gamma

        # Batch covariance calculation
        cov_values = cov_matrix[candidates][:, portfolio_items].mean(dim=1)
        return returns[candidates] / self.gamma - 0.5 * cov_values

    def forward(self,
                u: torch.Tensor,
                v: torch.Tensor,
                t: torch.Tensor,
                delta_t: torch.Tensor,
                edge_feat: torch.Tensor,
                returns: torch.Tensor,
                cov_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Batch memory updates
        u_emb = self.user_emb(u)
        v_emb = self.item_emb(v)

        # Vectorized message construction
        messages_u = torch.cat([
            u_emb,
            v_emb,
            delta_t.unsqueeze(1),
            edge_feat
        ], dim=1)

        # Update memories in batch
        for uid, msg in zip(u, messages_u):
            self.memory.update_memory(uid.item(), msg)

        # Batch attention computation
        z_u = self.attention(u_emb, v_emb.unsqueeze(1), edge_feat.unsqueeze(1))

        # Update portfolio using vectorized operations
        self.portfolio[u, v] = True

        # Batch candidate selection
        current_portfolio = self.portfolio[u]
        candidates = ~current_portfolio

        # Vectorized MV scoring
        mv_scores = self.compute_mv_scores(torch.arange(self.item_emb.num_embeddings),
                                           current_portfolio.nonzero()[:, 1],
                                           returns,
                                           cov_matrix)

        # Batch ranking operations
        pref_scores = torch.rand_like(mv_scores.float())
        combined_scores = self.lambda_mv * mv_scores + (1 - self.lambda_mv) * pref_scores

        # Select top items
        _, topk = torch.topk(combined_scores[candidates[u]], k=4, dim=1)
        pos_items = topk[:, 0]
        neg_items = topk[:, 1:4]

        return z_u, pos_items, neg_items


# Optimized training loop with mixed precision
def train(model: PfoTGNRec,
          data_loader: DataLoader,
          returns: torch.Tensor,
          cov_matrix: torch.Tensor,
          num_epochs: int = 20) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in data_loader:
            u, v, t, delta_t, edge_feat = batch
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                z_u, pos, negs = model(u, v, t, delta_t, edge_feat, returns, cov_matrix)
                pos_scores = torch.bmm(z_u.unsqueeze(1), model.item_emb(pos).unsqueeze(2)).squeeze()
                neg_scores = torch.bmm(z_u.unsqueeze(1), model.item_emb(negs).permute(0, 2, 1)).squeeze()

                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}")


# Example usage
if __name__ == "__main__":
    # Configuration
    num_users = 1000
    num_items = 500
    batch_size = 256
    seq_len = 30

    # Generate synthetic data
    users = torch.randint(0, num_users, (10000,))
    items = torch.randint(0, num_items, (10000,))
    timestamps = torch.cumsum(torch.randint(1, 10, (10000,)), dim=0)
    delta_times = torch.rand(10000)
    edge_feats = torch.rand(10000, 1)

    # Create dataset and loader
    dataset = TensorDataset(users, items, timestamps, delta_times, edge_feats)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    # Initialize model
    model = PfoTGNRec(num_users, num_items)
    returns = torch.randn(num_items)
    cov_matrix = torch.randn(num_items, num_items) @ torch.randn(num_items, num_items).T

    # Train
    train(model, data_loader, returns, cov_matrix)