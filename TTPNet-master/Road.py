import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Road(nn.Module):
    def __init__(self):
        super(Road, self).__init__()
        self.build()

    def build(self):
        self.embedding = nn.Embedding(128 * 128, 32)
        emb_vectors = np.load('Config/embedding_128.npy')
        self.embedding.weight.data.copy_(torch.from_numpy(emb_vectors))
        self.process_coords = nn.Linear(2 + 32, 32)

    def forward(self, traj):
        # Ensure `lngs`, `lats`, and `grid_id` are tensors
        if not isinstance(traj['lngs'], torch.Tensor):
            traj['lngs'] = torch.stack([torch.tensor(lng, dtype=torch.float32) for lng in traj['lngs']])
        if not isinstance(traj['lats'], torch.Tensor):
            traj['lats'] = torch.stack([torch.tensor(lat, dtype=torch.float32) for lat in traj['lats']])
        if not isinstance(traj['grid_id'], torch.Tensor):
            traj['grid_id'] = torch.stack([torch.tensor(grid, dtype=torch.long) for grid in traj['grid_id']])

        # Adjust all tensors to match the same sequence length
        max_len = min(traj['lngs'].size(1), traj['grid_id'].size(1))
        traj['lngs'] = traj['lngs'][:, :max_len]
        traj['lats'] = traj['lats'][:, :max_len]
        traj['grid_id'] = traj['grid_id'][:, :max_len]

        # Reshape for embedding
        lngs = torch.unsqueeze(traj['lngs'], dim=2)  # [batch_size, seq_len, 1]
        lats = torch.unsqueeze(traj['lats'], dim=2)  # [batch_size, seq_len, 1]
        grid_ids = torch.unsqueeze(traj['grid_id'], dim=2)  # [batch_size, seq_len, 1]

        # Get grid embeddings
        grids = self.embedding(grid_ids.squeeze(-1))  # [batch_size, seq_len, embedding_dim]

        # Concatenate lngs, lats, and grids
        locs = torch.cat([lngs, lats, grids], dim=2)  # [batch_size, seq_len, 2 + embedding_dim]
        locs = self.process_coords(locs)
        locs = torch.tanh(locs)

        return locs

