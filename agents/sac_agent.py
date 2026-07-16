# coding: utf-8
# =============================================================================
# sac_agent.py — Soft Actor-Critic agent for LUP genuine-cluster weighting.
#
# Replaces the discrete DQN with a continuous action α ∈ [0, 1] that
# controls the weighted combination of the two AHC clusters.
#
# Components:
#   • GaussianActor  — outputs μ, log_σ, samples via reparameterization
#   • QCritic        — clipped double-Q (two Q-networks)
#   • SACAgent       — full off-policy agent with automatic entropy tuning
#   • build_state_vector()       — same 12-dim state as dqn_agent.py
# =============================================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ──────────────────────── Replay Buffer ──────────────────────── #

class ReplayBuffer:
    """Fixed-size circular buffer for (S, A, R, S', done) transitions."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array([action], dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────── Gaussian Actor ──────────────────────── #

class GaussianActor(nn.Module):
    """
    Outputs a continuous action α ∈ [0, 1] via:
        μ, log_σ = MLP(state)
        z ~ N(μ, σ)      (reparameterized)
        a_raw = tanh(z)   ∈ [-1, 1]
        α = (a_raw + 1) / 2  ∈ [0, 1]
    """

    def __init__(self, state_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        h = self.shared(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """
        Returns
        -------
        action : tensor ∈ [0, 1]  — the trust weight α
        log_prob : tensor          — log π(a|s) with tanh + rescale correction
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        # Reparameterized sample
        z = dist.rsample()
        a_raw = torch.tanh(z)                     # ∈ [-1, 1]
        action = (a_raw + 1.0) / 2.0              # ∈ [0, 1]

        # Log-prob with tanh squashing correction + [0,1] rescale correction
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1.0 - a_raw.pow(2) + 1e-6)
        log_prob -= np.log(2.0)  # Jacobian of (x+1)/2
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state):
        """Deterministic action for evaluation (no noise)."""
        mu, _ = self.forward(state)
        return (torch.tanh(mu) + 1.0) / 2.0


# ──────────────────────── Q-Critic ──────────────────────── #

class QCritic(nn.Module):
    """
    Clipped double-Q: two independent Q(s, a) networks.
    Input: state (12-dim) concatenated with action (1-dim) → 13-dim.
    """

    def __init__(self, state_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        input_dim = state_dim + 1  # state + action

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ──────────────────────── SAC Agent ──────────────────────── #

class SACAgent:
    """
    Soft Actor-Critic with:
      • Continuous action α ∈ [0, 1]
      • Clipped double-Q critics
      • Automatic entropy coefficient (temperature) tuning
      • Soft target network updates (τ)

    Parameters
    ----------
    state_dim    : int   — state vector dimensionality  (12)
    hidden_dim   : int   — hidden layer width            (64)
    lr_actor     : float — actor learning rate            (3e-4)
    lr_critic    : float — critic learning rate           (3e-4)
    lr_alpha     : float — entropy coeff learning rate    (3e-4)
    gamma        : float — discount factor                (0.99)
    tau          : float — soft-update coefficient        (0.005)
    buffer_size  : int   — replay buffer capacity         (10_000)
    batch_size   : int   — minibatch size                 (64)
    init_alpha   : float — initial entropy coefficient    (0.2)
    target_entropy : float — target H (default: -dim(A)/2 = -0.5)
    device       : str   — 'cpu' or 'cuda'
    """

    def __init__(self, state_dim=12, hidden_dim=64,
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005,
                 buffer_size=10_000, batch_size=64,
                 init_alpha=0.2, target_entropy=None,
                 device='cpu'):

        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # ---- Networks ----
        self.actor = GaussianActor(state_dim, hidden_dim).to(self.device)
        self.critic = QCritic(state_dim, hidden_dim).to(self.device)
        self.critic_target = QCritic(state_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # ---- Optimizers ----
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # ---- Automatic entropy tuning ----
        self.target_entropy = target_entropy if target_entropy is not None else -0.5
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)

        # ---- Replay buffer ----
        self.memory = ReplayBuffer(capacity=buffer_size)

        # ---- Pending transition for delayed reward ----
        self._pending_state = None
        self._pending_action = None

    @property
    def alpha(self):
        """Current entropy coefficient (temperature)."""
        return self.log_alpha.exp().item()

    # ──────── Action selection ──────── #

    def select_action(self, state, deterministic=False):
        """
        Returns α ∈ [0, 1] — the trust weight for Cluster 1.

        Parameters
        ----------
        state : np.ndarray of shape (12,)
        deterministic : bool — if True, use mean (no sampling noise)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state_t)
            else:
                action, _ = self.actor.sample(state_t)
        return float(action.cpu().squeeze())

    # ──────── Transition storage ──────── #

    def store_transition(self, state, action, reward, next_state, done=False):
        self.memory.push(state, action, reward, next_state, done)

    def cache_pending(self, state, action):
        """Cache (S_t, A_t) while waiting for R_t."""
        self._pending_state = np.array(state, dtype=np.float32).copy()
        self._pending_action = float(action)

    def has_pending(self):
        return self._pending_state is not None

    def complete_pending(self, reward, next_state, done=False):
        """Finalize the pending transition and push to buffer."""
        if self._pending_state is None:
            return
        self.store_transition(self._pending_state, self._pending_action,
                              reward, next_state, done)
        self._pending_state = None
        self._pending_action = None

    # ──────── Network update ──────── #

    def update_model(self):
        """
        One gradient step on actor, critic, and entropy coefficient.
        Returns dict of losses for logging.
        """
        if len(self.memory) < self.batch_size:
            return {'critic_loss': 0.0, 'actor_loss': 0.0, 'alpha': self.alpha,
                    'alpha_loss': 0.0, 'alpha_grad_norm': 0.0}

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        alpha_val = self.log_alpha.exp().detach()

        # ──── Critic update ──── #
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(s2)
            q1_target, q2_target = self.critic_target(s2, next_action)
            q_target = torch.min(q1_target, q2_target)
            y = r + self.gamma * (1.0 - d) * (q_target - alpha_val * next_log_prob)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # ──── Actor update ──── #
        new_action, log_prob = self.actor.sample(s)
        q1_new = self.critic.q1_forward(s, new_action)
        actor_loss = (alpha_val * log_prob - q1_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        # ──── Entropy coefficient update ──── #
        alpha_loss = -(self.log_alpha.exp() *
                       (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # Capture gradient norm on log_alpha BEFORE the optimizer step
        alpha_grad_norm = float(torch.norm(self.log_alpha.grad).item())
        self.alpha_optim.step()

        # ──── Soft-update target critic ──── #
        self._soft_update()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
            'alpha_grad_norm': alpha_grad_norm,
        }

    def _soft_update(self):
        for tp, p in zip(self.critic_target.parameters(),
                         self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ──────── Persistence ──────── #

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optim': self.alpha_optim.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_optim.load_state_dict(ckpt['actor_optim'])
        self.critic_optim.load_state_dict(ckpt['critic_optim'])
        self.log_alpha = ckpt['log_alpha'].to(self.device).requires_grad_(True)
        self.alpha_optim.load_state_dict(ckpt['alpha_optim'])

# ──────────────────── State Builder ──────────────────── #

# Cache for computing temporal drift (ΔS) between rounds
_prev_semantic_state: dict = {'s_c1_cl': 0.5, 's_c1_cg': 0.5,
                               's_c2_cl': 0.5, 's_c2_cg': 0.5}


def build_state_vector(
    score_cluster_1, dev_1, absolute_deviation_1, kurt_1,
    score_cluster_2, dev_2, absolute_deviation_2, kurt_2,
    cluster_1_indices, cluster_2_indices,
    centroid_distance, ema_loss_trend, prev_alpha,
    # ── Semantic features (optional — None when semantic disabled) ──
    s_c1_cl=None, s_c1_cg=None,
    s_c2_cl=None, s_c2_cg=None,
):
    """
    Construct the state vector S_t.

    When semantic features are provided (not None), produces a 20-dim vector.
    When semantic features are None, produces the original 12-dim vector.

    Layout (20-dim):
        [0-3]   Cluster 1 geometric: norm_score, deviation, internal_sim, kurtosis
        [4-7]   Cluster 2 geometric: norm_score, deviation, internal_sim, kurtosis
        [8]     Cluster size ratio  |C1| / (|C1| + |C2|)
        [9]     Centroid L2 distance
        [10]    EMA loss trend
        [11]    Previous alpha value
        [12]    S_C1_cl:  Mean intra-client temporal MMD for Cluster 1
        [13]    S_C1_cg:  Mean client-vs-global MMD for Cluster 1
        [14]    S_C2_cl:  Mean intra-client temporal MMD for Cluster 2
        [15]    S_C2_cg:  Mean client-vs-global MMD for Cluster 2
        [16]    ΔS_C1_cl: Temporal drift of S_C1_cl (current - previous round)
        [17]    ΔS_C1_cg: Temporal drift of S_C1_cg
        [18]    ΔS_C2_cl: Temporal drift of S_C2_cl
        [19]    ΔS_C2_cg: Temporal drift of S_C2_cg
    """
    global _prev_semantic_state

    total_score = abs(score_cluster_1) + abs(score_cluster_2) + 1e-10
    total_size = len(cluster_1_indices) + len(cluster_2_indices) + 1e-10

    geometric = [
        score_cluster_1 / total_score,
        dev_1,
        absolute_deviation_1,
        kurt_1,
        score_cluster_2 / total_score,
        dev_2,
        absolute_deviation_2,
        kurt_2,
        len(cluster_1_indices) / total_size,
        centroid_distance,
        ema_loss_trend,
        float(prev_alpha),
    ]

    if s_c1_cl is not None:
        # Semantic features provided — build 20-dim state
        semantic_current = [s_c1_cl, s_c1_cg, s_c2_cl, s_c2_cg]

        # Temporal drift: ΔS = S(t) - S(t-1)
        delta_s = [
            s_c1_cl - _prev_semantic_state['s_c1_cl'],
            s_c1_cg - _prev_semantic_state['s_c1_cg'],
            s_c2_cl - _prev_semantic_state['s_c2_cl'],
            s_c2_cg - _prev_semantic_state['s_c2_cg'],
        ]

        # Update cache for next round
        _prev_semantic_state = {
            's_c1_cl': s_c1_cl, 's_c1_cg': s_c1_cg,
            's_c2_cl': s_c2_cl, 's_c2_cg': s_c2_cg,
        }

        state = np.array(
            geometric + semantic_current + delta_s, dtype=np.float32
        )
    else:
        # Semantic disabled — original 12-dim state
        state = np.array(geometric, dtype=np.float32)

    state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
    return state
