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
        self.hybrid_action = self.action_type in {"hybrid", "hybrid_dir_mag"}
        self.continuous_action = self.action_type in {
            "continuous",
            "continuous_tanh",
            "gaussian_tanh",
        }
        if self.hybrid_action:
            # Stored action layout: [exposure, dir_idx, magnitude]
            self.action_dim = 3
        else:
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
        self.policy_out_dim = 4 if self.hybrid_action else self.action_dim
        self.policy_head = nn.Linear(hidden, self.policy_out_dim)
        if self.continuous_action or self.hybrid_action:
            std_dim = 1 if self.hybrid_action else self.action_dim
            self.policy_log_std = nn.Parameter(
                torch.full((std_dim,), float(log_std_init), dtype=torch.float32)
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

    @staticmethod
    def _safe_logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = torch.clamp(x, eps, 1.0 - eps)
        return torch.log(x) - torch.log1p(-x)

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

    def _split_hybrid_policy(
        self,
        policy_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if policy_out.ndim != 2 or policy_out.shape[-1] < 4:
            raise ValueError(
                f"hybrid policy_out must be [B,4], got shape {tuple(policy_out.shape)}"
            )
        dir_logits = policy_out[:, :3]
        mag_mean = policy_out[:, 3:4]
        return dir_logits, mag_mean

    @staticmethod
    def _dir_idx_to_sign(dir_idx: torch.Tensor) -> torch.Tensor:
        # Class order: 0=flat, 1=long, 2=short.
        out = torch.zeros_like(dir_idx, dtype=torch.float32)
        out = torch.where(dir_idx == 1, torch.ones_like(out), out)
        out = torch.where(dir_idx == 2, -torch.ones_like(out), out)
        return out

    def _sample_hybrid(
        self,
        policy_out: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.policy_log_std is None:
            raise RuntimeError("Hybrid action requested without log-std parameter.")
        dir_logits, mag_mean = self._split_hybrid_policy(policy_out)
        dir_dist = torch.distributions.Categorical(logits=dir_logits)
        dir_idx = torch.argmax(dir_logits, dim=-1) if deterministic else dir_dist.sample()
        logp_dir = dir_dist.log_prob(dir_idx)
        ent_dir = dir_dist.entropy()

        std = torch.exp(self.policy_log_std).clamp(min=1e-4, max=5.0).view(1, -1)
        mag_dist = torch.distributions.Normal(mag_mean, std)
        pre_mag = mag_mean if deterministic else mag_dist.rsample()
        mag = torch.sigmoid(pre_mag)
        logp_mag = mag_dist.log_prob(pre_mag) - torch.log(mag * (1.0 - mag) + 1e-6)
        logp_mag = logp_mag.sum(dim=-1)
        ent_mag = mag_dist.entropy().sum(dim=-1)

        dir_sign = self._dir_idx_to_sign(dir_idx)
        exposure = dir_sign * mag.squeeze(-1)
        action = torch.stack(
            [exposure, dir_idx.to(dtype=torch.float32), mag.squeeze(-1)],
            dim=-1,
        )
        logp = logp_dir + logp_mag
        entropy = ent_dir + ent_mag
        return action, logp, entropy

    def _hybrid_actions_from_exposure(self, actions: torch.Tensor) -> torch.Tensor:
        exposure = actions.reshape(-1)
        mag = torch.clamp(exposure.abs(), 0.0, 1.0)
        dir_idx = torch.zeros_like(exposure, dtype=torch.float32)
        dir_idx = torch.where(exposure > 0.0, torch.ones_like(dir_idx), dir_idx)
        dir_idx = torch.where(exposure < 0.0, torch.full_like(dir_idx, 2.0), dir_idx)
        return torch.stack([exposure, dir_idx, mag], dim=-1)

    def _hybrid_logp_entropy(
        self,
        policy_out: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim == 1:
            actions = self._hybrid_actions_from_exposure(actions)
        elif actions.ndim == 2 and actions.shape[-1] == 1:
            actions = self._hybrid_actions_from_exposure(actions.squeeze(-1))
        elif actions.ndim != 2 or actions.shape[-1] < 3:
            raise ValueError(
                f"hybrid actions must be [B,3] or [B], got {tuple(actions.shape)}"
            )

        dir_logits, mag_mean = self._split_hybrid_policy(policy_out)
        dir_idx = actions[:, 1].round().to(dtype=torch.int64).clamp(min=0, max=2)
        mag = torch.clamp(actions[:, 2], 1e-6, 1.0 - 1e-6).unsqueeze(-1)

        dir_dist = torch.distributions.Categorical(logits=dir_logits)
        logp_dir = dir_dist.log_prob(dir_idx)
        ent_dir = dir_dist.entropy()

        if self.policy_log_std is None:
            raise RuntimeError("Hybrid action requested without log-std parameter.")
        std = torch.exp(self.policy_log_std).clamp(min=1e-4, max=5.0).view(1, -1)
        mag_dist = torch.distributions.Normal(mag_mean, std)
        pre_mag = self._safe_logit(mag)
        logp_mag = mag_dist.log_prob(pre_mag) - torch.log(mag * (1.0 - mag) + 1e-6)
        logp_mag = logp_mag.sum(dim=-1)
        ent_mag = mag_dist.entropy().sum(dim=-1)

        return logp_dir + logp_mag, ent_dir + ent_mag

    @torch.no_grad()
    def get_action_and_value(self, obs, device: torch.device | None = None):
        x = self._prepare_obs(obs, device)
        policy_out, value = self.forward(x)
        if self.hybrid_action:
            action, logp, entropy = self._sample_hybrid(policy_out, deterministic=False)
            return action, logp, entropy, value
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
        if self.hybrid_action:
            if torch.is_tensor(actions):
                a = actions.to(dtype=torch.float32, device=x.device)
            else:
                a = torch.as_tensor(actions, dtype=torch.float32, device=x.device)
            logp, entropy = self._hybrid_logp_entropy(policy_out, a)
            return logp, entropy, value
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
        if self.hybrid_action:
            action, logp, _entropy = self._sample_hybrid(policy_out, deterministic=False)
            if action.shape[0] == 1:
                return (
                    action.squeeze(0).detach().cpu().numpy().astype("float32"),
                    float(logp.item()),
                    float(value.item()),
                )
            return (
                action.detach().cpu().numpy().astype("float32"),
                logp.detach().cpu().numpy().astype("float32"),
                value.detach().cpu().numpy().astype("float32"),
            )
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
        if self.hybrid_action:
            actions, logp, _entropy = self._sample_hybrid(policy_out, deterministic=False)
            return actions, logp, value
        if self.continuous_action:
            actions, logp, _entropy = self._sample_continuous(policy_out, deterministic=False)
            return actions, logp, value
        dist = torch.distributions.Categorical(logits=policy_out)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp, value
