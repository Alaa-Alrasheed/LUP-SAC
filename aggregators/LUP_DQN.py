# coding: utf-8
# =============================================================================
# LUP_DQN.py — LUP aggregator with DQN-based genuine-cluster selection.
#
# This file is a SURGICAL MODIFICATION of LUP.py.  The following are
# preserved EXACTLY as in the original:
#   • MAD-based norm bounding (lines 93-149 of original)
#   • Feature extraction loop (lines 179-231 of original)
#   • Agglomerative Hierarchical Clustering (lines 240-250 of original)
#   • Norm-clipped final aggregation (lines 309-314 of original)
#
# WHAT CHANGED (clearly marked with >>>DQN>>> / <<<DQN<<< banners):
#   1. The first static genuine criterion (original lines 165-174) is
#      replaced by a DQN action that selects one of the two MAD-filtered
#      groups.
#   2. The second static genuine criterion (original lines 276-288) is
#      replaced by a DQN action that selects the honest cluster from AHC.
#   3. The aggregate() signature now accepts an optional `dqn_agent` and
#      `reward_calc` argument.
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import mstats
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_similarity
from .Centeredclipping import *
from .GeoMed import *

# Import DQN utilities — lives one directory up from aggregators/
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dqn_agent import build_state_vector

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

class LUP_DQN(object):
    """LUP aggregator with DQN-based cluster selection."""

    def __init__(self):
        self.name = "LUP_DQN"
        # Tracks which cluster was selected in the previous round (state feature).
        self._prev_selected_cluster: int = 0

    def aggregate(self, grads_original, user_grad_org_all, previous_grads,
                  score_matrix_client, f=10, epoch=1, g0=None, iteration=1,
                  dqn_agent=None, reward_calc=None, **kwargs):
        """
        Parameters
        ----------
        dqn_agent   : DQNAgent instance (from dqn_agent.py).  If None, falls
                      back to the original static heuristic so the code is
                      still runnable without RL.
        reward_calc : RewardCalculator instance.  Only used when dqn_agent
                      is provided.
        Everything else is identical to the original LUP.aggregate().
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
            # ==  STAGE 1: SELECT BETWEEN MAD-FILTERED vs OUTLIER GROUP  =====
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

            dev_1 = np.mean(dev_matrix_client_1[filtered_indices_list]) \
                if len(filtered_indices_list) > 0 else 0.0
            dev_1_v = np.mean(np.abs(set_1_grads_np - user_grad_org))
            dev_2 = np.mean(dev_matrix_client_1[filtered_indices_list_other]) \
                if len(filtered_indices_list_other) > 0 else 0.0
            dev_2_v = np.mean(np.abs(set_2_grads_np - user_grad_org))
            kurt_1_stage1 = np.mean(kurt_all[filtered_indices_list]) \
                if len(filtered_indices_list) > 0 else 0.0
            kurt_2_stage1 = np.mean(kurt_all[filtered_indices_list_other]) \
                if len(filtered_indices_list_other) > 0 else 0.0

            # >>>DQN>>>  STAGE-1 GENUINE CRITERION REPLACEMENT  >>>DQN>>>
            if dqn_agent is not None and len(filtered_indices_list_other) > 0:
                # Compute centroid L2 distance for stage 1
                centroid_dist_s1 = float(np.linalg.norm(
                    set_1_grads_np.flatten() - set_2_grads_np.flatten()))

                ema_trend = reward_calc.get_ema_loss_trend() if reward_calc else 0.0

                state_s1 = build_state_vector(
                    score_cluster_1=float(np.sum(
                        score_matrix_client[filtered_indices_list])),
                    dev_1=dev_1_v,
                    absolute_deviation_1=grad_sim_v_1,
                    kurt_1=kurt_1_stage1,
                    score_cluster_2=float(np.sum(
                        score_matrix_client[filtered_indices_list_other])),
                    dev_2=dev_2_v,
                    absolute_deviation_2=grad_sim_v_2,
                    kurt_2=kurt_2_stage1,
                    cluster_1_indices=filtered_indices_list,
                    cluster_2_indices=filtered_indices_list_other,
                    centroid_distance=centroid_dist_s1,
                    ema_loss_trend=ema_trend,
                    prev_selected_cluster=self._prev_selected_cluster,
                )

                action_s1 = dqn_agent.select_action(state_s1)
                if action_s1 == 1:  # select the "other" group
                    filtered_indices_list = filtered_indices_list_other
            else:
                # ---- ORIGINAL STATIC HEURISTIC (fallback) ---- #
                if (np.sum(score_matrix_client[filtered_indices_list_other])
                        >= np.sum(score_matrix_client[filtered_indices_list])):
                    if len(filtered_indices_list_other) != 0 and (
                        (grad_sim_v_2 > grad_sim_v_1 and kurt_2_stage1 > kurt_1_stage1 and dev_2_v < dev_1_v) or
                        (dev_2_v < dev_1_v and grad_sim_v_2 > grad_sim_v_1 and kurt_2_stage1 > kurt_1_stage1) or
                        (grad_sim_v_2 < grad_sim_v_1 and dev_2_v < dev_1_v) or
                        (grad_sim_v_2 > grad_sim_v_1 and dev_2_v < dev_1_v and kurt_2_stage1 > kurt_1_stage1)
                    ):
                        filtered_indices_list = filtered_indices_list_other
            # <<<DQN<<<  END STAGE-1 REPLACEMENT  <<<DQN<<<

            # ================================================================
            # ====  FEATURE EXTRACTION + AHC CLUSTERING (UNCHANGED)  =========
            # ================================================================
            features_list = []
            layer_grad = []
            list_layer_grad = []
            deviations = np.zeros([num_clients, 1])

            user_grad_org_all_filtered = [
                grads_original_copy[i] for i in filtered_indices_list]

            for i in range(len(user_grad_org_all_filtered)):
                clinet_index = filtered_indices_list[i]
                client_grad_i = user_grad_org_all_filtered[i].cpu().numpy().flatten()

                client_grad_i[np.isnan(client_grad_i)] = 0
                list_layer_grad.append(
                    user_grad_org_all_filtered[i].flatten())
                vector = client_grad_i

                layer_grad.append(list(client_grad_i))

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

            reduced_features = features_list

            if len(features_list) <= 1:
                cluster_1_indices = filtered_indices_list
                cluster_2_indices = filtered_indices_list
            else:
                agg_clustering = AgglomerativeClustering()
                cluster_labels = agg_clustering.fit_predict(reduced_features)
                value_indices = {
                    value: [index for index, val in enumerate(cluster_labels)
                            if val == value]
                    for value in set(cluster_labels)
                }
                cluster_1_indices = [
                    filtered_indices_list[i] for i in value_indices[0]]
                cluster_2_indices = [
                    filtered_indices_list[i] for i in value_indices[1]]

            # ================================================================
            # == STAGE 2: SELECT HONEST CLUSTER FROM AHC OUTPUT  =============
            # ================================================================
            score_cluster_1 = np.sum(score_matrix_client[cluster_1_indices])
            score_cluster_2 = np.sum(score_matrix_client[cluster_2_indices])

            absolute_deviation_1 = np.mean(grads_sim[cluster_1_indices])
            absolute_deviation_2 = np.mean(grads_sim[cluster_2_indices]) \
                if len(cluster_2_indices) > 0 else 0.0

            set_1_grads = grads_stacked[cluster_1_indices].mean(dim=0)
            set_2_grads = grads_stacked[cluster_2_indices].mean(dim=0) \
                if len(cluster_2_indices) > 0 else torch.zeros_like(grads_stacked[0])

            set_1_grads_np = torch.nan_to_num(
                set_1_grads.detach(), nan=0.0
            ).cpu().numpy().flatten().reshape(1, -1)
            set_2_grads_np = torch.nan_to_num(
                set_2_grads.detach(), nan=0.0
            ).cpu().numpy().flatten().reshape(1, -1)
            user_grad_org = torch.nan_to_num(
                user_grad_org_test_layer.detach(), nan=0.0
            ).cpu().numpy().flatten().reshape(1, -1)

            dev_1 = np.mean(np.abs(set_1_grads_np - user_grad_org))
            dev_2 = np.mean(np.abs(set_2_grads_np - user_grad_org))

            kurt_1 = np.mean(kurt_all[cluster_1_indices])
            kurt_2 = np.mean(kurt_all[cluster_2_indices]) \
                if len(cluster_2_indices) > 0 else 0.0

            filtered_indices_list_benign = cluster_1_indices  # default

            # >>>DQN>>>  STAGE-2 GENUINE CRITERION REPLACEMENT  >>>DQN>>>
            if dqn_agent is not None and len(cluster_2_indices) > 0:
                centroid_dist_s2 = float(np.linalg.norm(
                    set_1_grads_np.flatten() - set_2_grads_np.flatten()))

                ema_trend = reward_calc.get_ema_loss_trend() if reward_calc else 0.0

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
                    centroid_distance=centroid_dist_s2,
                    ema_loss_trend=ema_trend,
                    prev_selected_cluster=self._prev_selected_cluster,
                )

                action_s2 = dqn_agent.select_action(state_s2)

                if action_s2 == 1:
                    filtered_indices_list_benign = cluster_2_indices
                self._prev_selected_cluster = action_s2

                # Cache the pending transition for delayed-reward calculation
                dqn_agent.cache_pending(state_s2, action_s2)
            else:
                # ---- ORIGINAL STATIC HEURISTIC (fallback) ---- #
                if (len(cluster_2_indices) != 0
                        and score_cluster_2 >= score_cluster_1):
                    if (
                        (absolute_deviation_2 > absolute_deviation_1 and kurt_2 > kurt_1 and dev_2 < dev_1) or
                        (dev_2 < dev_1 and absolute_deviation_2 > absolute_deviation_1 and kurt_2 > kurt_1) or
                        (absolute_deviation_2 < absolute_deviation_1 and dev_2 < dev_1) or
                        (absolute_deviation_2 > absolute_deviation_1 and dev_2 < dev_1 and kurt_2 > kurt_1)
                    ):
                        filtered_indices_list_benign = cluster_2_indices
            # <<<DQN<<<  END STAGE-2 REPLACEMENT  <<<DQN<<<

            if len(cluster_1_indices) == 0:
                filtered_indices_list_benign = cluster_2_indices

            benign_indices = benign_indices + filtered_indices_list_benign
            normalized_deviation = 1 - (1 / (1 + deviations))

            dev_list = np.array(normalized_deviation.tolist())
            score_matrix_client[:, 0] += dev_list[:, 0]

            list_benign_indices.append(list(filtered_indices_list_benign))

        # ================================================================
        # ===========  FINAL AGGREGATION (UNCHANGED)  ====================
        # ================================================================
        benign_list = list(set(benign_indices))

        global_grad = grads_original
        selected_grads = global_grad[benign_list, :]

        score_matrix_client[benign_list] += 1

        grad_norm = torch.norm(selected_grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, max=norm_clip, out=None)
        grads_clip = (selected_grads / grad_norm) * grad_norm_clipped

        global_grad = grads_clip.mean(dim=0)

        return global_grad, benign_list, 0
