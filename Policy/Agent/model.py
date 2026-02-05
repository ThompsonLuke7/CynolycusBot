from __future__ import annotations
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        hidden: int = 128,
        head_mlp: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.head_mlp = bool(head_mlp)
        self.shared = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        if self.head_mlp:
            self.policy_mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.value_mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
        else:
            self.policy_mlp = nn.Identity()
            self.value_mlp = nn.Identity()
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(module.bias)
        if isinstance(self.policy_mlp, nn.Sequential):
            for module in self.policy_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
                    nn.init.zeros_(module.bias)
        if isinstance(self.value_mlp, nn.Sequential):
            for module in self.value_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _prepare_obs(self, obs, device: torch.device | None = None) -> torch.Tensor:
        if torch.is_tensor(obs):
            x = obs
        else:
            dev = device or next(self.parameters()).device
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"obs must be 1D or 2D, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs last dim must be {self.obs_dim}, got {x.shape[-1]}"
            )
        return x

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        p = self.policy_mlp(h)
        v = self.value_mlp(h)
        logits = self.policy_head(p)
        value = self.value_head(v).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def get_action_and_value(self, obs, device: torch.device | None = None):
        x = self._prepare_obs(obs, device)
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy, value

    def evaluate_actions(self, obs, actions):
        x = self._prepare_obs(obs)
        if torch.is_tensor(actions):
            a = actions
        else:
            a = torch.as_tensor(actions, dtype=torch.int64, device=x.device)
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        entropy = dist.entropy()
        return logp, entropy, value

    @torch.no_grad()
    def act(self, obs, device: torch.device):
        x = self._prepare_obs(obs, device)
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        if action.numel() == 1:
            return int(action.item()), float(logp.item()), float(value.item())
        return (
            action.detach().cpu().numpy().astype("int64"),
            logp.detach().cpu().numpy().astype("float32"),
            value.detach().cpu().numpy().astype("float32"),
        )

    @torch.no_grad()
    def act_batch(self, obs, device: torch.device):
        x = self._prepare_obs(obs, device)
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp, value
