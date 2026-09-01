# coding: utf-8
# =============================================================================
# LUP_SAC.py — LUP aggregator with SAC-based continuous cluster weighting.
#
# KEY DIFFERENCES vs. paper LUP:
#   • Pre-clustering heuristic (Stage 1, lines ~179-215): the ORIGINAL paper
#     static heuristic is preserved here, running BEFORE clustering to decide
#     which MAD-filtered half is the candidate-honest set.
#   • Post-clustering decision (Stage 2): instead of the paper's binary
#     genuine-criterion pick-one, the SAC agent outputs α ∈ [0,1] and the
#     global gradient is:
#         G = α · NormClip(Mean(C₁)) + (1-α) · NormClip(Mean(C₂))
#   • Clustering algorithm: replaced AHC with GMM + Cosine Sanity Check
#     (run_ensemble_filter in ensemble_filter.py).
#
# PRESERVED EXACTLY from original LUP.py:
#   • MAD-based norm bounding
#   • Feature extraction loop (7 statistical features)
#   • Pre-clustering static heuristic math (grad_sim / kurt / dev comparisons)
# =============================================================================
# NOTE: AgglomerativeClustering was removed; GMM is used instead via
# run_ensemble_filter(). See server/ensemble_filter.py for details.
# =============================================================================

from numpy.core.fromnumeric import partition
import tools
import math
import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
import time
import matplotlib.pyplot as plt
from itertools import cycle
import copy
import random
from itertools import product
from scipy.stats import skew, kurtosis
from scipy.stats import iqr, median_abs_deviation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
# AgglomerativeClustering removed — GMM + Cosine Sanity Check is used instead
# (see run_ensemble_filter in server/ensemble_filter.py)
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import mstats
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_similarity


# Import SAC utilities
import sys, os
from .ensemble_filter import run_ensemble_filter
from agents.sac_agent import build_state_vector

# ---------------------------------------------------------------------------- #
# Entropy helpers (unchanged from LUP.py)

def calculate_entropy(Yi, L):
    d = (np.max(Yi) - np.min(Yi)) / L
    bins = [np.min(Yi) + d * j for j in range(L+1)]
    mj = np.histogram(Yi, bins)[0]
    pj = mj / len(Yi)
    entropy = -np.sum([p * np.log2(p) for p in pj if p > 0])
    return entropy

def get_entropy_and_info_gain(X, L):
    K, N = X.shape
    H_total = 0
    entropy = []
    for i in range(K):
        Yi = X[i]
        Hi = calculate_entropy(Yi, L)
        entropy.append(Hi)
        H_total += Hi
    H = H_total / K
    information_gains = []
    for k in range(K):
        concatenated_data = []
        for i in range(K):
            if i != k:
                concatenated_data.extend(X[i])
        Hk = H - calculate_entropy(np.array(concatenated_data), L)
        information_gains.append(Hk)
    return information_gains, entropy


# ---------------------------------------------------------------------------- #

class LUP_SAC(object):
    """LUP aggregator with SAC-based continuous cluster weighting."""

    def __init__(self):
        self.name = "LUP_SAC"
        self._prev_alpha: float = 0.5  # previous round's α

    def aggregate(self, grads_original, user_grad_org_all, previous_grads,
                  score_matrix_client, f=10, epoch=1, g0=None, iteration=1,
                  sac_agent=None, reward_calc=None, semantic_analyzer=None,
                  args=None, **kwargs):
        """
        Parameters
        ----------
        sac_agent         : SACAgent instance.  If None, falls back to original
                            static heuristic.
        reward_calc       : SmoothnessRewardCalculator instance.
        semantic_analyzer : SemanticDataDefense instance (if --semantic is True)
        """

        grads_original_copy = grads_original
        grads_original = torch.stack(grads_original, dim=0)
        num_clients = int(len(user_grad_org_all))
        benign_indices = []
        list_benign_indices = []
        replacement_value = 0.0

        if True:
            # ================================================================
            # ============  NORM BOUNDING USING MAD (UNCHANGED)  =============
            # ================================================================
            all_clients_layer_grad = grads_original
            grads_stacked = all_clients_layer_grad
            grads_stacked[torch.isnan(grads_stacked)] = 0

            grad_l2norm = torch.norm(grads_stacked, dim=1)
            median_population = torch.median(grad_l2norm)
            deviations = torch.abs(grad_l2norm - median_population)
            scaling_factor = torch.median(deviations)
            upper_bound = median_population + scaling_factor
            lower_bound = median_population - scaling_factor

            filtered_indices = torch.where(
                (grad_l2norm >= lower_bound) & (grad_l2norm <= upper_bound))

            user_grad_org_test_layer = previous_grads
            dev_matrix_client_1 = np.zeros([num_clients, 1])
            kurt_all = np.zeros([num_clients, 1])
            skew_all = np.zeros([num_clients, 1])

            grads_dists = tools.pairwise_distance_faster(grads_original)
            dist_score = grads_dists[:, :num_clients // 2].sum(dim=1)
            grads_sim = dist_score.squeeze(dim=-1)
            grads_sim = grads_sim.squeeze(dim=-1).cpu().numpy()

            for i in range(len(all_clients_layer_grad)):
                client_grad_i = grads_original[i].cpu().numpy().flatten()
                kurt_all[i, 0] = np.std(client_grad_i)
                skew_all[i, 0] = np.abs(skew(client_grad_i))
                user_grad_org = torch.nan_to_num(
                    user_grad_org_test_layer.detach(), nan=0.0
                ).cpu().numpy().flatten()
                user_grad_org = user_grad_org.reshape(1, -1)
                client_grad_i = client_grad_i.reshape(1, -1)
                dev_matrix_client_1[i, 0] = np.mean(
                    np.abs(client_grad_i - user_grad_org))

            filtered_indices_list = filtered_indices[0].tolist()
            filtered_indices_list_other = list(
                set(range(num_clients)) - set(filtered_indices_list))

            # ================================================================
            # ============  SEMANTIC ANALYSIS (IF ENABLED)  ==================
            # ================================================================
            per_client_semantic_scores = {}
            if semantic_analyzer is not None:
                # We only need to analyze clients that passed MAD filtering
                # (or we could analyze all of them; for efficiency, we analyze all
                # clients because GMM clustering uses them all later).
                # Flattened gradients:
                client_grads_flattened = []
                for i in range(num_clients):
                    client_grads_flattened.append(grads_original[i].cpu().numpy().flatten())
                
                # To interface with semantic_analyzer, we need torch tensors.
                client_grads_tensors = [torch.tensor(g, dtype=torch.float32) for g in client_grads_flattened]
                client_indices_list = list(range(num_clients))
                
                per_client_semantic_scores = semantic_analyzer.analyze(
                    client_grads=client_grads_tensors,
                    client_indices=client_indices_list,
                    global_model=semantic_analyzer._global_model  # passed in main
                )

            # ================================================================
            # ==  STAGE 1: ORIGINAL STATIC HEURISTIC (no RL here)  ===========
            # ================================================================
            grad_sim_v_2 = (np.mean(grads_sim[filtered_indices_list_other])
                            if len(filtered_indices_list_other) > 0 else 0.0)
            grad_sim_v_1 = (np.mean(grads_sim[filtered_indices_list])
                            if len(filtered_indices_list) > 0 else 0.0)

            set_1_grads = grads_stacked[filtered_indices_list].mean(dim=0) \
                if len(filtered_indices_list) > 0 else torch.zeros_like(grads_stacked[0])
            set_2_grads = grads_stacked[filtered_indices_list_other].mean(dim=0) \
                if len(filtered_indices_list_other) > 0 else torch.zeros_like(grads_stacked[0])
            set_1_grads_np = torch.nan_to_num(
                set_1_grads.detach(), nan=0.0).cpu().numpy().flatten().reshape(1, -1)
            set_2_grads_np = torch.nan_to_num(
                set_2_grads.detach(), nan=0.0).cpu().numpy().flatten().reshape(1, -1)
            user_grad_org = torch.nan_to_num(
                user_grad_org_test_layer.detach(), nan=0.0
            ).cpu().numpy().flatten().reshape(1, -1)

            dev_1_v = np.mean(np.abs(set_1_grads_np - user_grad_org))
            dev_2_v = np.mean(np.abs(set_2_grads_np - user_grad_org))
            kurt_1_stage1 = np.mean(kurt_all[filtered_indices_list]) \
                if len(filtered_indices_list) > 0 else 0.0
            kurt_2_stage1 = np.mean(kurt_all[filtered_indices_list_other]) \
                if len(filtered_indices_list_other) > 0 else 0.0

            # ORIGINAL STATIC HEURISTIC for Stage 1
            if (np.sum(score_matrix_client[filtered_indices_list_other])
                    >= np.sum(score_matrix_client[filtered_indices_list])):
                if len(filtered_indices_list_other) != 0 and (
                    (grad_sim_v_2 > grad_sim_v_1 and kurt_2_stage1 > kurt_1_stage1 and dev_2_v < dev_1_v) or
                    (dev_2_v < dev_1_v and grad_sim_v_2 > grad_sim_v_1 and kurt_2_stage1 > kurt_1_stage1) or
                    (grad_sim_v_2 < grad_sim_v_1 and dev_2_v < dev_1_v) or
                    (grad_sim_v_2 > grad_sim_v_1 and dev_2_v < dev_1_v and kurt_2_stage1 > kurt_1_stage1)
                ):
                    filtered_indices_list = filtered_indices_list_other

            # ================================================================
            # ====  FEATURE EXTRACTION + GMM CLUSTERING  =====================
            # ====  (Enhanced with directional features for Label-Flip /   ===
            # ====   Sign-Flip isolation)                                  ===
            # ====  Clustering delegated to run_ensemble_filter()          ===
            # ================================================================
            features_list = []
            layer_grad = []
            list_layer_grad = []
            deviations = np.zeros([num_clients, 1])

            user_grad_org_all_filtered = [
                grads_original_copy[i] for i in filtered_indices_list]

            # ── Per-client statistical features (original) ──
            for i in range(len(user_grad_org_all_filtered)):
                clinet_index = filtered_indices_list[i]
                client_grad_i = user_grad_org_all_filtered[i].cpu().numpy().flatten()

                client_grad_i[np.isnan(client_grad_i)] = 0
                list_layer_grad.append(
                    user_grad_org_all_filtered[i].flatten())
                vector = client_grad_i

                layer_grad.append(client_grad_i.copy())

                positive_count = np.sum(vector > 0)
                negative_count = np.sum(vector < 0)
                zero_count = np.sum(vector == 0)

                median_value = np.median(vector)
                skewness = skew(vector)
                kurt = kurtosis(vector)

                norm_v = np.linalg.norm(vector)

                absolute_deviation = np.mean(np.abs(
                    vector - user_grad_org_test_layer.detach().cpu().numpy()))
                user_grad_org_test_layer[np.isnan(client_grad_i)] = 0
                dir = np.arccos(np.dot(
                    vector, user_grad_org_test_layer.detach().cpu().numpy()
                ) / (np.linalg.norm(vector) *
                     np.linalg.norm(user_grad_org_test_layer.detach().cpu().numpy())))
                deviations[clinet_index, 0] = (
                    deviations[clinet_index, 0] + absolute_deviation)

                vector_features = [
                    positive_count, negative_count, zero_count,
                    kurt, skewness, absolute_deviation, norm_v
                ]

                tensor_data = torch.tensor(vector_features)
                replacement_value = 0.0
                tensor_data[torch.isnan(tensor_data)] = replacement_value
                tensor_data[torch.isinf(tensor_data)] = 1
                vector_features = tensor_data.tolist()

                features_list.append(vector_features)

            # ── Directional Feature 1: Historical Cosine Similarity ──
            # Each client's gradient vs the server's EMA of global gradients.
            # Label-Flip / Sign-Flip grads diverge from the EMA direction,
            # so this feature will be low for attackers.
            ema_grad = None
            if reward_calc is not None and hasattr(reward_calc, '_ema_grad'):
                ema_grad = reward_calc._ema_grad

            historical_cos_sims = []
            for i in range(len(layer_grad)):
                vec = layer_grad[i]
                if ema_grad is not None:
                    dot = np.dot(vec, ema_grad)
                    norm_a = np.linalg.norm(vec) + 1e-10
                    norm_b = np.linalg.norm(ema_grad) + 1e-10
                    cos_sim = float(dot / (norm_a * norm_b))
                else:
                    cos_sim = 0.0
                historical_cos_sims.append(
                    float(np.nan_to_num(cos_sim, nan=0.0,
                                        posinf=1.0, neginf=-1.0)))

            # ── Directional Feature 2: Pairwise Angular Distance ──
            # Average cosine distance (1 - cos_sim) between this client's
            # gradient and every other client's gradient this round.
            # Attackers will have high angular distance from the benign
            # majority, making them easy to cluster separately.
            n_filtered = len(layer_grad)
            pairwise_ang_dists = []
            for i in range(n_filtered):
                total_dist = 0.0
                count = 0
                for j in range(n_filtered):
                    if i != j:
                        dot = np.dot(layer_grad[i], layer_grad[j])
                        ni = np.linalg.norm(layer_grad[i]) + 1e-10
                        nj = np.linalg.norm(layer_grad[j]) + 1e-10
                        sim = float(dot / (ni * nj))
                        total_dist += (1.0 - sim)  # cosine distance
                        count += 1
                avg_dist = total_dist / max(count, 1)
                pairwise_ang_dists.append(
                    float(np.nan_to_num(avg_dist, nan=0.0,
                                        posinf=2.0, neginf=0.0)))

            # ── Inject directional features into feature vectors ──
            for i in range(len(features_list)):
                features_list[i].append(historical_cos_sims[i])
                features_list[i].append(pairwise_ang_dists[i])

            # ── Normalize all features for GMM ──
            # MinMaxScaler bounds magnitude features to [0, 1] without
            # shifting the origin, preserving angular geometry for Euclidean.
            if len(features_list) > 1:
                scaler = MinMaxScaler(feature_range=(0, 1))
                reduced_features = scaler.fit_transform(features_list)

                reduced_features = np.nan_to_num(
                    reduced_features, nan=0.0, posinf=1.0, neginf=-1.0
                ).tolist()
            else:
                reduced_features = features_list

            if len(features_list) <= 1:
                cluster_1_indices = filtered_indices_list
                cluster_2_indices = filtered_indices_list
            else:
                cluster_1_indices, cluster_2_indices = run_ensemble_filter(
                    features_list, historical_cos_sims, filtered_indices_list
                )

            # ================================================================
            # == >>>SAC>>> STAGE 2: CONTINUOUS WEIGHTED AGGREGATION >>>SAC>>> =
            # ================================================================
            score_cluster_1 = np.sum(score_matrix_client[cluster_1_indices])
            score_cluster_2 = np.sum(score_matrix_client[cluster_2_indices])

            # ── Aggregate Semantic Scores (if enabled) ──
            s_c1_cl = s_c1_cg = s_c2_cl = s_c2_cg = None
            if semantic_analyzer is not None:
                s_c1_cl, s_c1_cg = semantic_analyzer.aggregate_cluster_scores(
                    per_client_semantic_scores, cluster_1_indices
                )
                s_c2_cl, s_c2_cg = semantic_analyzer.aggregate_cluster_scores(
                    per_client_semantic_scores, cluster_2_indices
                )
                
                # Register with reward calculator for R_dist
                if reward_calc is not None:
                    # We register both, but we don't know which is "selected" yet
                    # The reward_calc uses this when computing reward.
                    # We will register them such that the one with higher SAC weight
                    # will be the 'selected' one. We do this *after* SAC selection.
                    pass

            absolute_deviation_1 = np.mean(grads_sim[cluster_1_indices])
            absolute_deviation_2 = np.mean(grads_sim[cluster_2_indices]) \
                if len(cluster_2_indices) > 0 else 0.0

            c1_mean = grads_stacked[cluster_1_indices].mean(dim=0) \
                if len(cluster_1_indices) > 0 else torch.zeros_like(grads_stacked[0])
            c2_mean = grads_stacked[cluster_2_indices].mean(dim=0) \
                if len(cluster_2_indices) > 0 else torch.zeros_like(grads_stacked[0])

            c1_mean_np = torch.nan_to_num(
                c1_mean.detach(), nan=0.0).cpu().numpy().flatten().reshape(1, -1)
            c2_mean_np = torch.nan_to_num(
                c2_mean.detach(), nan=0.0).cpu().numpy().flatten().reshape(1, -1)
            user_grad_org = torch.nan_to_num(
                user_grad_org_test_layer.detach(), nan=0.0
            ).cpu().numpy().flatten().reshape(1, -1)

            dev_1 = np.mean(np.abs(c1_mean_np - user_grad_org))
            dev_2 = np.mean(np.abs(c2_mean_np - user_grad_org))

            kurt_1 = np.mean(kurt_all[cluster_1_indices])
            kurt_2 = np.mean(kurt_all[cluster_2_indices]) \
                if len(cluster_2_indices) > 0 else 0.0

            # ── Per-cluster direction & magnitude diagnostics ──
            ema_ref = reward_calc._ema_grad if (reward_calc is not None
                      and hasattr(reward_calc, '_ema_grad')
                      and reward_calc._ema_grad is not None) else None
            if ema_ref is not None:
                _c1_flat = c1_mean_np.flatten()
                _c2_flat = c2_mean_np.flatten()
                _ema_norm = float(np.linalg.norm(ema_ref)) + 1e-10
                c1_dir_score = float(
                    np.dot(_c1_flat, ema_ref)
                    / ((np.linalg.norm(_c1_flat) + 1e-10) * _ema_norm))
                c2_dir_score = float(
                    np.dot(_c2_flat, ema_ref)
                    / ((np.linalg.norm(_c2_flat) + 1e-10) * _ema_norm))
                c1_mag_ratio = float(np.linalg.norm(_c1_flat) / _ema_norm)
                c2_mag_ratio = float(np.linalg.norm(_c2_flat) / _ema_norm)
            else:
                c1_dir_score = 0.0
                c2_dir_score = 0.0
                c1_mag_ratio = 1.0
                c2_mag_ratio = 1.0

            # -- Determine alpha (continuous weight) --
            alpha_val = 1.0  # default: full weight on C1

            if sac_agent is not None and len(cluster_2_indices) > 0:
                centroid_dist = float(np.linalg.norm(
                    c1_mean_np.flatten() - c2_mean_np.flatten()))

                ema_trend = (reward_calc.get_ema_loss_trend()
                             if reward_calc else 0.0)

                state_s2 = build_state_vector(
                    score_cluster_1=float(score_cluster_1),
                    dev_1=dev_1,
                    absolute_deviation_1=absolute_deviation_1,
                    kurt_1=kurt_1,
                    score_cluster_2=float(score_cluster_2),
                    dev_2=dev_2,
                    absolute_deviation_2=absolute_deviation_2,
                    kurt_2=kurt_2,
                    cluster_1_indices=cluster_1_indices,
                    cluster_2_indices=cluster_2_indices,
                    centroid_distance=centroid_dist,
                    ema_loss_trend=ema_trend,
                    prev_alpha=self._prev_alpha,
                    s_c1_cl=s_c1_cl, s_c1_cg=s_c1_cg,
                    s_c2_cl=s_c2_cl, s_c2_cg=s_c2_cg,
                )

                alpha_val = sac_agent.select_action(state_s2)
                alpha_pre_clip = alpha_val  # Raw SAC output before any clipping

                # ================================================================
                #  BURN-IN PHASE: Override SAC during early epochs
                # ================================================================
                if epoch <= 5:
                    if score_cluster_1 >= score_cluster_2:
                        alpha_val = 1.0
                    else:
                        alpha_val = 0.0
                else:
                    # ================================================================
                    #  ALPHA CLIPPING: Binarize high-confidence weights
                    # ================================================================
                    if alpha_val > 0.85:
                        alpha_val = 1.0
                    elif alpha_val < 0.15:
                        alpha_val = 0.0

                # ── Synchronous safety gate (semantic + direction + magnitude) ──
                sync_gate_fired = False
                r_dist_prospective = 0.0
                gate_info = {'fired': False, 'semantic': False,
                             'direction': False, 'magnitude': False,
                             'c1_dir_score': c1_dir_score, 'c2_dir_score': c2_dir_score,
                             'c1_mag_ratio': c1_mag_ratio, 'c2_mag_ratio': c2_mag_ratio,
                             'alpha_pre_clip': alpha_pre_clip}
                if (semantic_analyzer is not None and reward_calc is not None
                        and s_c1_cg is not None and s_c2_cg is not None):
                    alpha_val, r_dist_prospective, gate_info = \
                        reward_calc.evaluate_synchronous_gate(
                            s_c1_cg, s_c2_cg,
                            c1_dir_score, c2_dir_score,
                            c1_mag_ratio, c2_mag_ratio,
                            alpha_val, self._prev_alpha, args)
                    sync_gate_fired = gate_info['fired']
                    # Carry raw diagnostic scores through for telemetry
                    gate_info['c1_dir_score'] = c1_dir_score
                    gate_info['c2_dir_score'] = c2_dir_score
                    gate_info['c1_mag_ratio'] = c1_mag_ratio
                    gate_info['c2_mag_ratio'] = c2_mag_ratio

                self._prev_alpha = alpha_val

                # Cache for delayed reward
                sac_agent.cache_pending(state_s2, alpha_val)
            else:
                # FALLBACK: original static heuristic for binary selection
                alpha_pre_clip = alpha_val  # No SAC in fallback; raw = final
                sync_gate_fired = False
                r_dist_prospective = 0.0
                gate_info = {'fired': False, 'semantic': False,
                             'direction': False, 'magnitude': False,
                             'c1_dir_score': c1_dir_score, 'c2_dir_score': c2_dir_score,
                             'c1_mag_ratio': c1_mag_ratio, 'c2_mag_ratio': c2_mag_ratio,
                             'alpha_pre_clip': alpha_pre_clip}
                if (len(cluster_2_indices) != 0
                        and score_cluster_2 >= score_cluster_1):
                    if (
                        (absolute_deviation_2 > absolute_deviation_1 and kurt_2 > kurt_1 and dev_2 < dev_1) or
                        (dev_2 < dev_1 and absolute_deviation_2 > absolute_deviation_1 and kurt_2 > kurt_1) or
                        (absolute_deviation_2 < absolute_deviation_1 and dev_2 < dev_1) or
                        (absolute_deviation_2 > absolute_deviation_1 and dev_2 < dev_1 and kurt_2 > kurt_1)
                    ):
                        alpha_val = 0.0  # full weight on C2

                # ── Synchronous safety gate (fallback branch) ──
                if (semantic_analyzer is not None and reward_calc is not None
                        and s_c1_cg is not None and s_c2_cg is not None):
                    alpha_val, r_dist_prospective, gate_info = \
                        reward_calc.evaluate_synchronous_gate(
                            s_c1_cg, s_c2_cg,
                            c1_dir_score, c2_dir_score,
                            c1_mag_ratio, c2_mag_ratio,
                            alpha_val, self._prev_alpha, args)
                    sync_gate_fired = gate_info['fired']
                    gate_info['c1_dir_score'] = c1_dir_score
                    gate_info['c2_dir_score'] = c2_dir_score
                    gate_info['c1_mag_ratio'] = c1_mag_ratio
                    gate_info['c2_mag_ratio'] = c2_mag_ratio

            # -- Norm-clip each cluster mean independently --
            c1_norm = torch.norm(c1_mean)
            c2_norm = torch.norm(c2_mean)
            clip_val = min(c1_norm.item(), c2_norm.item()) + 1e-10
            c1_clipped = c1_mean * min(1.0, clip_val / (c1_norm.item() + 1e-10))
            c2_clipped = c2_mean * min(1.0, clip_val / (c2_norm.item() + 1e-10))

            # -- Weighted aggregation: G = α·C₁ + (1-α)·C₂ --
            global_grad = alpha_val * c1_clipped + (1.0 - alpha_val) * c2_clipped

            if len(cluster_1_indices) == 0:
                global_grad = c2_clipped
            # <<<SAC<<<  END STAGE-2  <<<SAC<<<

            # -- Register Reputation Score for R_rep computation --
            if reward_calc is not None:
                participating_indices = cluster_1_indices + cluster_2_indices
                if len(participating_indices) > 0:
                    mean_rs = np.mean(score_matrix_client[participating_indices])
                    max_rs = np.max(score_matrix_client)
                    r_rep = float(mean_rs / (max_rs + 1e-5))
                    reward_calc.register_reputation(r_rep)
                    
                # -- Register Semantic Scores for R_dist computation --
                if semantic_analyzer is not None:
                    # alpha_val > 0.5 means C1 is more highly weighted
                    if alpha_val >= 0.5:
                        s_sel_cg = s_c1_cg
                        s_rej_cg = s_c2_cg
                    else:
                        s_sel_cg = s_c2_cg
                        s_rej_cg = s_c1_cg
                    
                    reward_calc.register_semantic_scores(s_sel_cg, s_rej_cg)

            # -- All indices contribute (for score tracking) --
            all_participating = list(set(cluster_1_indices + cluster_2_indices))
            benign_indices = benign_indices + all_participating

            normalized_deviation = 1 - (1 / (1 + deviations))
            dev_list = np.array(normalized_deviation.tolist())
            score_matrix_client[:, 0] += dev_list[:, 0]

            list_benign_indices.append(all_participating)

        # ================================================================
        # ===========  FINAL OUTPUT  =====================================
        # ================================================================
        benign_list = list(set(benign_indices))
        score_matrix_client[benign_list] += 1

        # global_grad is already computed as the weighted combination above
        return global_grad, benign_list, alpha_val, sync_gate_fired, r_dist_prospective, gate_info
