# coding: utf-8
# =============================================================================
# semantic_analyzer.py — Semantic Distribution Defense for LUP-SAC.
#
# Extracts gradient-inversion-based semantic features per client,
# computes MMD distribution similarity, and tracks temporal stability.
#
# Outputs per-client scores:
#   S_k_cl  — MMD(client_previous_feature, client_current_feature)
#   S_k_cg  — MMD(client_current_feature, global_feature)
#
# These are aggregated to cluster-level in aggregation.py before
# being fed to the SAC state vector.
#
# Architectural Notes:
#   • The SAC agent remains the SOLE decision-maker.
#   • This module NEVER accepts or rejects clients directly.
#   • When --semantic is False, this module is never instantiated
#     (zero-overhead short-circuit).
#
# Adapted from AdaAggRL (Wang et al., AAAI 2025):
#   • Gradient inversion logic from utilities.GradientReconstructor
#   • MMD computation from utilities.maximum_mean_discrepancy
#   • Feature extraction pattern from exp_environments.py
#
# Key differences from AdaAggRL:
#   • TV penalty bypassed for tabular/MLP data (L2 regularization)
#   • Auto-scaling GI iterations based on device (GPU vs CPU)
#   • Client sampling with temporal decay on stale scores
#   • Pretrained feature extractor separate from FL model
#   • Dataset-agnostic API (same interface for image & tabular)
# =============================================================================

from __future__ import annotations

import copy
import os
import random
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ═══════════════════════════════════════════════════════════════════════
#  MMD Computation  (extracted from AdaAggRL/utilities.py)
# ═══════════════════════════════════════════════════════════════════════

def compute_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared pairwise Euclidean distances between x and y.

    Args:
        x: tensor of shape [num_x_samples, num_features]
        y: tensor of shape [num_y_samples, num_features]

    Returns:
        Distance matrix of shape [num_x_samples, num_y_samples].
    """
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda t: torch.sum(torch.square(t), dim=1)
    return torch.transpose(
        norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1
    )


def gaussian_kernel_matrix(
    x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor
) -> torch.Tensor:
    """Multi-scale Gaussian RBF kernel.

    Args:
        x: tensor [num_samples_x, num_features]
        y: tensor [num_samples_y, num_features]
        sigmas: tensor of kernel bandwidths

    Returns:
        Kernel matrix [num_samples_x, num_samples_y].
    """
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1.0 / (2.0 * sigmas)
    dist = compute_pairwise_distances(x, y).float()
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))
    return torch.reshape(torch.sum(torch.exp(-s), 0), dist.shape)


def mmd_raw(
    x: torch.Tensor, y: torch.Tensor,
    kernel=gaussian_kernel_matrix,
) -> torch.Tensor:
    """Raw MMD² statistic: E[K(x,x)] + E[K(y,y)] - 2E[K(x,y)]."""
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    return cost


_MMD_SIGMAS = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6,
]


def maximum_mean_discrepancy(
    source: torch.Tensor, target: torch.Tensor
) -> float:
    """Compute bounded MMD between two feature sets.

    Uses a multi-scale Gaussian RBF kernel and normalizes via
    ``2 * cos(tanh(0.5 * MMD²)) - 1`` to bound output to ~[-1, 1].

    Higher values (closer to 1) indicate greater similarity.
    Lower values indicate distributional divergence.

    Args:
        source: [N, D] feature tensor (CPU)
        target: [M, D] feature tensor (CPU)

    Returns:
        Scalar MMD similarity score (float).
    """
    sigmas_var = Variable(torch.FloatTensor(_MMD_SIGMAS))
    kernel_fn = partial(gaussian_kernel_matrix, sigmas=sigmas_var)
    cost = mmd_raw(source, target, kernel=kernel_fn)
    if cost < 0:
        cost = torch.tensor(0.0)
    cost = 2 * torch.cos(torch.tanh(0.5 * cost)) - 1
    return float(cost.item())


# ═══════════════════════════════════════════════════════════════════════
#  Gradient Reconstruction  (adapted from AdaAggRL/utilities.py)
# ═══════════════════════════════════════════════════════════════════════

def _total_variation_image(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV for 4D image tensors (B, C, H, W)."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def _l2_regularization(x: torch.Tensor) -> torch.Tensor:
    """L2 norm penalty for tabular reconstructions.

    Unlike TV which exploits spatial smoothness, tabular features have no
    adjacency structure.  L2 penalizes extreme reconstructed values,
    acting as a Gaussian prior on feature space.
    """
    return torch.mean(x ** 2)


def _reconstruction_costs(
    gradients: List[Tuple[torch.Tensor, ...]],
    input_gradient: List[torch.Tensor],
    cost_fn: str = 'sim',
    indices: str = 'def',
    weights: str = 'equal',
) -> torch.Tensor:
    """Gradient matching cost (cosine similarity by default).

    Operates on per-layer gradient tuples — fully model-agnostic.
    """
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    else:
        indices = torch.arange(len(input_gradient))

    ex = input_gradient[0]
    weight_vals = ex.new_ones(len(input_gradient))

    total_costs = 0.0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0.0
        offset = 0
        for i in indices:
            i = int(i)
            ti = i - offset
            if ti >= len(trial_gradient) or ti < 0:
                continue
            if input_gradient[i].shape != trial_gradient[ti].shape:
                offset += 1
                continue
            if cost_fn == 'sim':
                costs -= (trial_gradient[ti] * input_gradient[i]).sum() * weight_vals[i]
                pnorm[0] += trial_gradient[ti].pow(2).sum() * weight_vals[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weight_vals[i]
            else:  # l2 fallback
                costs += (
                    (trial_gradient[ti] - input_gradient[i]).pow(2)
                ).sum() * weight_vals[i]
        if cost_fn == 'sim':
            costs = 1 + costs / (pnorm[0].sqrt() * pnorm[1].sqrt() + 1e-10)
        total_costs += costs
    return total_costs / max(len(gradients), 1)


class _GradientInverter:
    """Lightweight gradient inversion engine.

    Reconstructs a dummy batch of data from per-layer gradients.
    Used only to approximate the client's local data distribution,
    NOT for pixel-accurate reconstruction.

    Args:
        model: The global model (used for forward/backward passes).
        dm: Data mean (scalar or tuple).
        ds: Data std  (scalar or tuple).
        is_image: Whether data is spatial (image) or tabular.
        max_iterations: Optimization steps for reconstruction.
        dummy_batch_size: Number of dummy samples to reconstruct.
        device: torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        dm: float,
        ds: float,
        is_image: bool,
        max_iterations: int,
        dummy_batch_size: int,
        device: torch.device,
    ):
        self.model = model
        self.dm = dm
        self.ds = ds
        self.is_image = is_image
        self.max_iterations = max_iterations
        self.dummy_batch_size = dummy_batch_size
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _init_dummy(self, input_shape: Tuple[int, ...]) -> torch.Tensor:
        """Initialize dummy input (zeros — fast convergence)."""
        return torch.zeros(
            (self.dummy_batch_size, *input_shape),
            device=self.device,
            dtype=torch.float32,
        )

    def reconstruct(
        self,
        input_gradient: List[torch.Tensor],
        input_shape: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, float]:
        """Reconstruct dummy data from gradients.

        Args:
            input_gradient: List of per-layer gradient tensors.
            input_shape: Shape of a single input sample (e.g., (1,28,28)
                         for MNIST or (10,) for ToN-IoT).

        Returns:
            (reconstructed_data, reconstruction_loss)
        """
        self.model.eval()
        self.model.zero_grad()

        x_trial = self._init_dummy(input_shape)
        x_trial.requires_grad = True

        # Recover labels using iDLG trick (multi-image generalization)
        try:
            last_weight_min = torch.argmin(
                torch.sum(input_gradient[-2], dim=-1), dim=-1
            )
            labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
            # Tile labels to match batch size
            labels = labels.expand(self.dummy_batch_size)
            reconstruct_label = False
        except Exception:
            reconstruct_label = True
            labels = None

        if reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(
                (self.dummy_batch_size, output_test.shape[1]),
                device=self.device,
            ).requires_grad_(True)
            optimizer = torch.optim.Adam([x_trial, labels], lr=0.1)
            loss_fn = lambda pred, lbl: torch.mean(
                torch.sum(
                    -F.softmax(lbl, dim=-1) * F.log_softmax(pred, dim=-1), 1
                )
            )
        else:
            optimizer = torch.optim.Adam([x_trial], lr=0.1)
            loss_fn = self.loss_fn

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.max_iterations * 0.375),
                int(self.max_iterations * 0.625),
                int(self.max_iterations * 0.875),
            ],
            gamma=0.1,
        )

        best_loss = float('inf')
        best_x = x_trial.detach().clone()

        for iteration in range(self.max_iterations):
            def closure():
                optimizer.zero_grad()
                self.model.zero_grad()
                pred = self.model(x_trial)
                loss = loss_fn(pred, labels)
                gradient = torch.autograd.grad(
                    loss, self.model.parameters(), create_graph=True
                )
                rec_loss = _reconstruction_costs(
                    [gradient], input_gradient, cost_fn='sim'
                )
                # Regularization
                if self.is_image:
                    rec_loss += 1e-6 * _total_variation_image(x_trial)
                else:
                    rec_loss += 1e-2 * _l2_regularization(x_trial)
                rec_loss.backward()
                return rec_loss

            rec_loss = optimizer.step(closure)
            scheduler.step()

            with torch.no_grad():
                # Projection
                if self.is_image:
                    dm_t = torch.as_tensor(
                        self.dm, device=self.device, dtype=torch.float32
                    )
                    ds_t = torch.as_tensor(
                        self.ds, device=self.device, dtype=torch.float32
                    )
                    if dm_t.dim() == 0:
                        dm_t = dm_t.view(1, 1, 1)
                        ds_t = ds_t.view(1, 1, 1)
                    x_trial.data = torch.clamp(
                        x_trial.data,
                        min=float((-dm_t / ds_t).min()),
                        max=float(((1.0 - dm_t) / ds_t).max()),
                    )
                # Track best
                loss_val = float(rec_loss.item()) if hasattr(rec_loss, 'item') else float(rec_loss)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_x = x_trial.detach().clone()

        return best_x, best_loss


# ═══════════════════════════════════════════════════════════════════════
#  Pretrained Feature Extractors
# ═══════════════════════════════════════════════════════════════════════

class _ImageFeatureExtractor(nn.Module):
    """Lightweight CNN feature extractor for image datasets (MNIST/FMNIST/CIFAR).

    Uses a small pretrained CNN.  The last classification layer is removed,
    exposing the penultimate representation.

    This is a SEPARATE model from the FL global model — it is frozen and
    never participates in federated training.
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5), # features.0
            nn.ReLU(),                       # features.1
            nn.MaxPool2d(2),                 # features.2
            nn.Conv2d(16, 32, kernel_size=3),# features.3
            nn.ReLU(),                       # features.4
            nn.MaxPool2d(2),                 # features.5
            nn.Conv2d(32, 64, kernel_size=3) # features.6
        )
        self._feature_dim = 64 * 3 * 3  # Based on 28x28 MNIST input

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).view(x.size(0), -1)


class _TabularFeatureExtractor(nn.Module):
    """Lightweight MLP feature extractor for tabular datasets (ToN-IoT).

    Maps raw tabular features through a small MLP to a learned
    representation space.  The classification head is removed.

    This is a SEPARATE model from the FL global model.
    """

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self._feature_dim = 32

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _build_feature_extractor(
    dataset: str,
    device: torch.device,
    train_loader=None,
) -> nn.Module:
    """Factory: build and self-pretrain a feature extractor.

    For image datasets, uses a small CNN.
    For tabular datasets, uses a small MLP.

    The extractor is trained for a few epochs on a sample of the
    training data to learn meaningful representations, then frozen.

    Args:
        dataset: Name of the dataset ('mnist', 'fmnist', 'cifar', 'ton_iot').
        device: Torch device.
        train_loader: A DataLoader for pretraining (uses first client's data
                      or test data).  If None, extractor is used with random
                      weights (still useful for distributional comparison).

    Returns:
        Frozen nn.Module with a .feature_dim attribute.
    """
    if dataset in ('ton_iot',):
        extractor_core = _TabularFeatureExtractor(input_dim=10)
        num_classes = 10
    elif dataset in ('cifar',):
        extractor_core = _ImageFeatureExtractor(input_channels=3, num_classes=10)
        num_classes = 10
    else:  # mnist, fmnist
        extractor_core = _ImageFeatureExtractor(input_channels=1, num_classes=10)
        num_classes = 10

    pretrained_path = f"pretrained_{dataset}_extractor.pth"
    if os.path.exists(pretrained_path):
        print(f"  [SemanticAnalyzer] Loading pretrained frozen extractor from {pretrained_path}")
        extractor_core.load_state_dict(torch.load(pretrained_path, map_location=device), strict=True)
    else:
        # Loud Error: Strict enforcement to avoid falling back to untrained weights
        raise FileNotFoundError(
            f"\n\n[FATAL ERROR] Pre-trained feature extractor not found at '{pretrained_path}'.\n"
            f"The Semantic Analyzer requires a decoupled, frozen feature extractor to prevent early-epoch noise.\n"
            f"Please run the standalone utility script first:\n"
            f"  python train_extractor.py\n\n"
            f"(If using Tabular dataset 'ton_iot', ensure a pre-trained Autoencoder/MLP is saved at 'pretrained_ton_iot_extractor.pth')\n"
        )

    # The extractor must never train or update.
    extractor_core.eval()
    for param in extractor_core.parameters():
        param.requires_grad = False

    return extractor_core.to(device)


# ═══════════════════════════════════════════════════════════════════════
#  Main Defense Class
# ═══════════════════════════════════════════════════════════════════════

class SemanticDataDefense:
    """Semantic distribution defense module for LUP-SAC.

    Performs gradient inversion on client updates, extracts features from
    reconstructed data, computes MMD-based distribution similarity scores,
    and tracks temporal stability.

    This module is a PASSIVE feature provider.  It outputs per-client
    semantic scores that are aggregated to cluster-level and appended to
    the SAC state vector.  It NEVER accepts or rejects clients.

    Args:
        dataset: Name of the dataset ('mnist', 'fmnist', 'cifar', 'ton_iot').
        global_model: The FL global model (used for gradient inversion).
        device: Torch device.
        gi_iterations: Max gradient inversion optimization steps.
            If -1, auto-scaled based on device (GPU: 30, CPU: 15).
        gi_batch_size: Number of dummy samples per inversion.
        sample_ratio: Fraction of clients to analyze per round.
        decay_factor: Temporal decay applied to stale (unsampled) scores.
            Score_stale = Score_prev * decay_factor.
        train_loader: DataLoader for pretraining the feature extractor.
    """

    def __init__(
        self,
        dataset: str,
        global_model: nn.Module,
        device: torch.device,
        gi_iterations: int = -1,
        gi_batch_size: int = 8,
        sample_ratio: float = 0.3,
        decay_factor: float = 0.9,
        train_loader=None,
    ):
        self.dataset = dataset
        self.device = device
        self.gi_batch_size = gi_batch_size
        self.sample_ratio = sample_ratio
        self.decay_factor = decay_factor
        self.is_image = dataset not in ('ton_iot',)

        # Auto-scale GI iterations based on device
        if gi_iterations == -1:
            if device.type == 'cuda':
                self.gi_iterations = 30
            else:
                self.gi_iterations = 15
        else:
            self.gi_iterations = gi_iterations

        # Determine input shape
        if dataset == 'ton_iot':
            self._input_shape: Tuple[int, ...] = (10,)
        elif dataset == 'cifar':
            self._input_shape = (3, 32, 32)
        else:  # mnist, fmnist
            self._input_shape = (1, 28, 28)

        # Data statistics for image bounding
        if dataset == 'ton_iot':
            self._dm = 0.0
            self._ds = 1.0
        elif dataset == 'cifar':
            self._dm = 0.5
            self._ds = 0.5
        else:
            self._dm = 0.1307
            self._ds = 0.3081

        # Build pretrained feature extractor (separate from FL model)
        self.extractor = _build_feature_extractor(
            dataset, device, train_loader
        )

        # Store reference to global model for gradient inversion
        self._global_model = global_model

        # Per-client historical state
        # key: client_index (int), value: dict
        self._client_history: Dict[int, dict] = {}

        # Global feature proxy distribution (computed once from clean proxy data)
        self._global_feature: Optional[torch.Tensor] = None
        if train_loader is not None:
            self._compute_proxy_global_feature(train_loader)

        # Per-client scores from last round (for unsampled clients)
        self._cached_scores: Dict[int, Tuple[float, float]] = {}

    def _compute_proxy_global_feature(self, train_loader) -> None:
        """Compute the Global Data Distribution using clean proxy data."""
        print("  [SemanticAnalyzer] Computing global data distribution from clean proxy data...")
        self.extractor.eval()
        features_list = []
        max_samples = 1000  # Cap the number of proxy samples for MMD performance
        total_samples = 0
        with torch.no_grad():
            for images, _ in train_loader:
                feats = self.extractor(images.to(self.device))
                features_list.append(feats.cpu())
                total_samples += feats.shape[0]
                if total_samples >= max_samples:
                    break
        if features_list:
            self._global_feature = torch.cat(features_list, dim=0)[:max_samples]
            print(f"  [SemanticAnalyzer] Global feature proxy computed. Shape: {self._global_feature.shape}")

    def _get_or_init_client(self, client_idx: int) -> dict:
        """Get client history entry, creating it if needed."""
        if client_idx not in self._client_history:
            self._client_history[client_idx] = {
                'times': 0,
                'local_feature': None,  # Previous round's feature
                'current_feature': None,
            }
        return self._client_history[client_idx]

    def _invert_gradient(
        self,
        client_grad: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, float]:
        """Perform gradient inversion for a single client.

        Args:
            client_grad: Flattened gradient tensor for this client.
            model: The global model (to compute forward/backward).

        Returns:
            (reconstructed_data, reconstruction_loss)
        """
        # Convert flattened gradient to per-layer format
        per_layer_grads: List[torch.Tensor] = []
        cur_pos = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                layer_grad = client_grad[cur_pos:cur_pos + numel].view(
                    param.shape
                ).clone().detach().to(self.device)
                per_layer_grads.append(layer_grad)
                cur_pos += numel

        inverter = _GradientInverter(
            model=model,
            dm=self._dm,
            ds=self._ds,
            is_image=self.is_image,
            max_iterations=self.gi_iterations,
            dummy_batch_size=self.gi_batch_size,
            device=self.device,
        )

        reconstructed, rec_loss = inverter.reconstruct(
            per_layer_grads, self._input_shape
        )
        return reconstructed, rec_loss

    def _extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract features from (reconstructed) data using the pretrained
        feature extractor.

        Args:
            data: Tensor of shape (B, *input_shape).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        with torch.no_grad():
            features = self.extractor(data.to(self.device))
        return features.detach()

    def analyze(
        self,
        client_grads: List[torch.Tensor],
        client_indices: List[int],
        global_model: nn.Module,
    ) -> Dict[int, Tuple[float, float]]:
        """Analyze a set of client gradients and produce semantic scores.

        This is the main entry point called from aggregation.py.

        Strategy:
            1. Sample a subset of clients for full GI analysis.
            2. For unsampled clients, reuse previous scores with decay.
            3. Compute per-client MMD scores.
            4. Update historical state.

        Args:
            client_grads: List of flattened gradient tensors, one per client.
            client_indices: List of global client indices (matching grads).
            global_model: Current global model (for GI forward/backward).

        Returns:
            Dict mapping client_index → (S_cl, S_cg) where:
                S_cl: MMD(previous_feature, current_feature)  — temporal self-stability
                S_cg: MMD(current_feature, global_feature)    — global alignment
        """
        num_clients = len(client_grads)
        if num_clients == 0:
            return {}

        # ── Step 1: Select clients to analyze ──
        num_to_sample = max(1, int(num_clients * self.sample_ratio))
        if num_to_sample >= num_clients:
            sampled_local_indices = list(range(num_clients))
        else:
            sampled_local_indices = sorted(
                random.sample(range(num_clients), num_to_sample)
            )

        # ── Step 2: Gradient Inversion + Feature Extraction ──
        model_copy = copy.deepcopy(global_model)
        model_copy.to(self.device)
        model_copy.eval()

        analyzed_features: Dict[int, torch.Tensor] = {}

        for local_idx in sampled_local_indices:
            global_idx = client_indices[local_idx]
            grad = client_grads[local_idx]

            try:
                reconstructed, rec_loss = self._invert_gradient(
                    grad, model_copy
                )
                features = self._extract_features(reconstructed)
                analyzed_features[global_idx] = features
            except Exception as e:
                # If inversion fails, skip this client (reuse stale score)
                print(f"  [SemanticAnalyzer] GI failed for client {global_idx}: {e}")
                continue

        # ── Step 3: (Removed) Global distribution is now precomputed from proxy data ──

        # ── Step 4: Compute per-client MMD scores ──
        scores: Dict[int, Tuple[float, float]] = {}

        for local_idx in range(num_clients):
            global_idx = client_indices[local_idx]
            client_state = self._get_or_init_client(global_idx)

            if global_idx in analyzed_features:
                # Full analysis available
                current_feature = analyzed_features[global_idx]
                client_state['current_feature'] = current_feature.cpu()

                # S_cl: temporal self-stability
                if client_state['local_feature'] is not None:
                    s_cl = maximum_mean_discrepancy(
                        client_state['local_feature'],
                        current_feature.cpu(),
                    )
                else:
                    # First time — optimistic initialization
                    s_cl = 0.9

                # S_cg: global alignment
                if self._global_feature is not None:
                    s_cg = maximum_mean_discrepancy(
                        current_feature.cpu(),
                        self._global_feature.cpu(),
                    )
                else:
                    s_cg = 0.9

                scores[global_idx] = (s_cl, s_cg)
                self._cached_scores[global_idx] = (s_cl, s_cg)

                # Update history
                client_state['local_feature'] = current_feature.cpu().clone()
                client_state['times'] += 1

            else:
                # Unsampled client — reuse with temporal decay
                if global_idx in self._cached_scores:
                    prev_cl, prev_cg = self._cached_scores[global_idx]
                    # Decay toward neutral (0.5) over time
                    decayed_cl = prev_cl * self.decay_factor + 0.5 * (
                        1 - self.decay_factor
                    )
                    decayed_cg = prev_cg * self.decay_factor + 0.5 * (
                        1 - self.decay_factor
                    )
                    scores[global_idx] = (decayed_cl, decayed_cg)
                    self._cached_scores[global_idx] = (decayed_cl, decayed_cg)
                else:
                    # Never analyzed — neutral score
                    scores[global_idx] = (0.5, 0.5)

        # Cleanup model copy
        del model_copy

        return scores

    @staticmethod
    def aggregate_cluster_scores(
        per_client_scores: Dict[int, Tuple[float, float]],
        cluster_indices: List[int],
    ) -> Tuple[float, float]:
        """Aggregate per-client semantic scores to cluster-level.

        Computes mean S_cl and mean S_cg for the given cluster.

        Args:
            per_client_scores: Dict mapping client_idx → (S_cl, S_cg).
            cluster_indices: List of client indices in this cluster.

        Returns:
            (mean_S_cl, mean_S_cg) for the cluster.
        """
        if not cluster_indices:
            return 0.5, 0.5

        s_cls = []
        s_cgs = []
        for idx in cluster_indices:
            if idx in per_client_scores:
                cl, cg = per_client_scores[idx]
                s_cls.append(cl)
                s_cgs.append(cg)
            else:
                s_cls.append(0.5)
                s_cgs.append(0.5)

        return float(np.mean(s_cls)), float(np.mean(s_cgs))
