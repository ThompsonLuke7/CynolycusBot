from __future__ import annotations
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        action_type: str = "discrete",
        action_dim: int = 1,
        log_std_init: float = -0.5,
        hidden: int = 128,
        head_mlp: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_type = str(action_type).strip().lower()
        self.continuous_action = self.action_type in {
            "continuous",
            "continuous_tanh",
            "gaussian_tanh",
        }
        self.action_dim = int(action_dim if self.continuous_action else n_actions)
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
        self.policy_head = nn.Linear(hidden, self.action_dim)
        if self.continuous_action:
            self.policy_log_std = nn.Parameter(
                torch.full((self.action_dim,), float(log_std_init), dtype=torch.float32)
            )
        else:
            self.policy_log_std = None
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
        policy_out = self.policy_head(p)
        value = self.value_head(v).squeeze(-1)
        return policy_out, value

    @staticmethod
    def _safe_atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _continuous_dist(self, mean: torch.Tensor) -> torch.distributions.Normal:
        if self.policy_log_std is None:
            raise RuntimeError("Continuous action requested without log-std parameter.")
        std = torch.exp(self.policy_log_std).clamp(min=1e-4, max=5.0)
        return torch.distributions.Normal(mean, std)

    def _sample_continuous(
        self,
        mean: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._continuous_dist(mean)
        pre_tanh = mean if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        logp = logp.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, logp, entropy

    def _continuous_logp_entropy(
        self,
        mean: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"continuous actions last dim must be {self.action_dim}, got {actions.shape[-1]}"
            )
        dist = self._continuous_dist(mean)
        pre_tanh = self._safe_atanh(actions)
        logp = dist.log_prob(pre_tanh) - torch.log(1.0 - actions.pow(2) + 1e-6)
        logp = logp.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy

    @torch.no_grad()
    def get_action_and_value(self, obs, device: torch.device | None = None):
        x = self._prepare_obs(obs, device)
        policy_out, value = self.forward(x)
        if self.continuous_action:
            action, logp, entropy = self._sample_continuous(policy_out, deterministic=False)
            return action, logp, entropy, value
        dist = torch.distributions.Categorical(logits=policy_out)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy, value

    def evaluate_actions(self, obs, actions):
        x = self._prepare_obs(obs)
        policy_out, value = self.forward(x)
        if self.continuous_action:
            if torch.is_tensor(actions):
                a = actions.to(dtype=torch.float32, device=x.device)
            else:
                a = torch.as_tensor(actions, dtype=torch.float32, device=x.device)
            logp, entropy = self._continuous_logp_entropy(policy_out, a)
            return logp, entropy, value
        if torch.is_tensor(actions):
            a = actions.to(dtype=torch.int64, device=x.device)
        else:
            a = torch.as_tensor(actions, dtype=torch.int64, device=x.device)
        dist = torch.distributions.Categorical(logits=policy_out)
        logp = dist.log_prob(a)
        entropy = dist.entropy()
        return logp, entropy, value

    @torch.no_grad()
    def act(self, obs, device: torch.device):
        x = self._prepare_obs(obs, device)
        policy_out, value = self.forward(x)
        if self.continuous_action:
            action, logp, _entropy = self._sample_continuous(policy_out, deterministic=False)
            if action.numel() == 1:
                return float(action.item()), float(logp.item()), float(value.item())
            return (
                action.detach().cpu().numpy().astype("float32"),
                logp.detach().cpu().numpy().astype("float32"),
                value.detach().cpu().numpy().astype("float32"),
            )
        dist = torch.distributions.Categorical(logits=policy_out)
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
        policy_out, value = self.forward(x)
        if self.continuous_action:
            actions, logp, _entropy = self._sample_continuous(policy_out, deterministic=False)
            return actions, logp, value
        dist = torch.distributions.Categorical(logits=policy_out)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp, value
