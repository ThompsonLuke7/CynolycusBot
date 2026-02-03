from __future__ import annotations
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 3, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs, device: torch.device):
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    @torch.no_grad()
    def act_batch(self, obs, device: torch.device):
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        return (
            actions.detach().cpu().numpy().astype("int64"),
            logp.detach().cpu().numpy().astype("float32"),
            value.detach().cpu().numpy().astype("float32"),
        )
