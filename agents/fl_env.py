# coding: utf-8
import numpy as np

# ──────────────── Composite Reward Calculator (Anti-Hijacking) ──────────────── #

class CompositeRewardCalculator:
    """
    Hardened composite reward to prevent EMA momentum hijacking.

    R_t = R_dir + R_mag + R_rep

    Components
    ----------
    R_dir (Direction):
        CosineSimilarity(G_current, G_previous_EMA).
        Range: [-1, +1].

    R_mag (Magnitude Penalty):
        If ||G_current||₂ > 2 × ||G_previous_EMA||₂ → R_mag = -1.0
        Otherwise → R_mag = 0.0
        Catches Sign-Flip and similar attacks that cause L2 norm explosions.

    R_rep (Reputation Anchor):
        Mean(RS_selected_clients) / (Max(RS_all_clients) + 1e-5).
        Replaces R_var to defeat ByzMean's geometric cloaking.

    Total reward bounded roughly in [-1.0, +2.0].

    The EMA uses a slow decay (α=0.1 by default) so attackers cannot
    corrupt the reference direction within a few rounds.
    """

    def __init__(self, ema_alpha: float = 0.1,
                 mag_threshold_factor: float = 2.0,
                 mag_penalty: float = -1.0):
        """
        Parameters
        ----------
        ema_alpha : float
            Weight for new gradient in EMA update.  Kept low (0.1) to
            resist poisoning — attacker needs ~10 consecutive hijacks
            to significantly shift the reference.
        mag_threshold_factor : float
            If ||G||₂ > factor × ||EMA||₂, fire R_mag penalty.
        mag_penalty : float
            Flat penalty value when magnitude gate fires.
        """
        self.ema_alpha = ema_alpha
        self.mag_threshold_factor = mag_threshold_factor
        self.mag_penalty = mag_penalty

        self._ema_grad = None
        self._loss_history = []
        
        # Cached per-round: set by register_reputation() before compute_reward()
        self._r_rep = None

    # ── Loss tracking (kept for state builder compatibility) ──

    def update_loss_history(self, loss: float):
        self._loss_history.append(loss)

    def get_ema_loss_trend(self) -> float:
        """EMA trend over last 3 losses. Positive = loss rising."""
        if len(self._loss_history) < 2:
            return 0.0
        window = self._loss_history[-3:]
        diffs = [window[i+1] - window[i] for i in range(len(window)-1)]
        return float(np.mean(diffs))

    # ── EMA gradient tracking ──

    def update_ema_grad(self, global_grad):
        if self._ema_grad is None:
            self._ema_grad = global_grad.copy()
        else:
            self._ema_grad = (self.ema_alpha * global_grad
                              + (1.0 - self.ema_alpha) * self._ema_grad)

    # ── Per-round reputation registration ──

    def register_reputation(self, r_rep: float):
        """
        Call this each round with the calculated reputation score.
        """
        self._r_rep = r_rep

    # ── Reward computation ──

    def compute_reward(self, global_grad):
        """
        R_t = R_dir + R_mag + R_rep

        Parameters
        ----------
        global_grad : np.ndarray
            Flattened global gradient from current round.

        Returns
        -------
        (reward, cos_sim, r_dir, r_mag, r_rep) : tuple
            reward  — composite reward (float)
            cos_sim — raw cosine similarity for logging (float)
            r_dir   — directional reward component (float)
            r_mag   — magnitude penalty component (float)
            r_rep   — reputation anchor component (float)
        """
        # ─── R_dir: Directional momentum ───
        if self._ema_grad is None:
            cos_sim = 0.0
        else:
            dot = np.dot(global_grad, self._ema_grad)
            norm_a = np.linalg.norm(global_grad) + 1e-10
            norm_b = np.linalg.norm(self._ema_grad) + 1e-10
            cos_sim = float(dot / (norm_a * norm_b))
        r_dir = cos_sim

        # ─── R_mag: Magnitude explosion penalty ───
        r_mag = 0.0
        if self._ema_grad is not None:
            ema_norm = float(np.linalg.norm(self._ema_grad))
            cur_norm = float(np.linalg.norm(global_grad))
            if ema_norm > 1e-10 and cur_norm > self.mag_threshold_factor * ema_norm:
                r_mag = self.mag_penalty  # -1.0

        # ─── R_rep: Reputation Anchor ───
        r_rep = self._r_rep if self._r_rep is not None else 0.0

        # ─── Composite reward ───
        reward = r_dir + r_mag + r_rep

        # Clear cached reputation
        self._r_rep = None

        return reward, cos_sim, r_dir, r_mag, r_rep


# Backward-compatible alias so existing imports still work
SmoothnessRewardCalculator = CompositeRewardCalculator
