# coding: utf-8

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
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import iqr, median_abs_deviation
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import  mstats
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_similarity
from .Centeredclipping import * 
from .GeoMed import *

# ---------------------------------------------------------------------------- #
# LGP detector, using clustering to remove outlier

def calculate_entropy(Yi, L):
    d = (np.max(Yi) - np.min(Yi)) / L
    bins = [np.min(Yi) + d * j for j in range(L+1)]
    mj = np.histogram(Yi, bins)[0]
    pj = mj / len(Yi)
    entropy = -np.sum([p * np.log2(p) for p in pj if p > 0])
    return entropy

def get_entropy_and_info_gain(X, L):
    K, N = X.shape  # Get the shape of X as K x N
   
    H_total = 0

    # Calculate entropy for each worker
    entropy = []
    for i in range(K):  # Iterate over K workers
        Yi = X[i]
        Hi = calculate_entropy(Yi, L)
        entropy.append(Hi)
        H_total += Hi
    
    H = H_total / K
    
    # Calculate information gain for each worker
    information_gains = []
    
    for k in range(K):  # Iterate over N samples
        concatenated_data = []
        for i in range(K):  # Iterate over K workers
            if i != k:
                concatenated_data.extend(X[i])

        
        Hk = H - calculate_entropy(np.array(concatenated_data), L)
        information_gains.append(Hk)

    return information_gains,entropy


class LUP(object):
    def __init__(self):
        self.name = "LGP"
    def aggregate(self, grads_original,user_grad_org_all, previous_grads,score_matrix_client,f=10, epoch=1, g0=None, iteration=1, **kwargs):
        

            grads_original_copy = grads_original
            grads_original = torch.stack(grads_original, dim=0)   
            num_clients = int(len(user_grad_org_all))
            benign_indices = []
            list_benign_indices =   []            
            replacement_value = 0.0
            
            if True:

                #_______________________________Norm Bounding Using MAD__________________________________________

                all_clients_layer_grad = grads_original

                grads_stacked= all_clients_layer_grad


                grads_stacked[torch.isnan(grads_stacked)] = 0 # remove nan

                # gradient norm-based clustering
                grad_l2norm = torch.norm(grads_stacked, dim=1)
                # Step 1: Calculate the median of the population
                median_population = torch.median(grad_l2norm)

                # Step 2: Calculate the deviations from the median
                deviations = torch.abs(grad_l2norm - median_population)

                # Step 3: Find the median of the absolute deviations (scaling factor)
                scaling_factor = torch.median(deviations) 
 
                
                # Step 4: Calculate the upper bound
                upper_bound = median_population +  scaling_factor 

                # Step 5: Calculate the lower bound
                lower_bound = median_population -   scaling_factor 

                filtered_indices = torch.where( (grad_l2norm >= lower_bound) &  (grad_l2norm <= upper_bound))  #  

                user_grad_org_test_layer = previous_grads
                dev_matrix_client_1 = np.zeros([num_clients,1])
                kurt_all =  np.zeros([num_clients,1])
                skew_all =  np.zeros([num_clients,1])
                
                grads_dists = tools.pairwise_distance_faster(grads_original)
                dist_score = grads_dists[:,:num_clients//2].sum(dim=1)
                grads_sim = dist_score.squeeze(dim=-1)
                grads_sim = grads_sim.squeeze(dim=-1).cpu().numpy()
                
                
                for i in range(len(all_clients_layer_grad)):
                    client_grad_i= grads_original[i].cpu().numpy().flatten()
                    kurt_all[i,0] = np.std(client_grad_i)
                    skew_all[i,0] = np.abs(skew(client_grad_i))
                    user_grad_org = torch.nan_to_num(user_grad_org_test_layer.detach(), nan=0.0).cpu().numpy().flatten()
                    user_grad_org = user_grad_org.reshape(1,-1)
                    client_grad_i = client_grad_i.reshape(1,-1)
                    dev_matrix_client_1[i,0] = np.mean(np.abs(client_grad_i - user_grad_org))
                    

                filtered_indices_list = filtered_indices[0].tolist()
                filtered_indices_list_other = list(set(range(50)) - set(filtered_indices_list))
                     

                grad_sim_v_2 = np.mean(grads_sim[filtered_indices_list_other]) 
                grad_sim_v_1 = np.mean(grads_sim[filtered_indices_list]) 


                set_1_grads = grads_stacked[filtered_indices_list].mean(dim=0)
                set_2_grads = grads_stacked[filtered_indices_list_other].mean(dim=0)
                set_1_grads = torch.nan_to_num(set_1_grads.detach(), nan=0.0).cpu().numpy().flatten().reshape(1,-1)
                set_2_grads = torch.nan_to_num(set_2_grads.detach(), nan=0.0).cpu().numpy().flatten().reshape(1,-1)
                user_grad_org = torch.nan_to_num(user_grad_org_test_layer.detach(), nan=0.0).cpu().numpy().flatten().reshape(1,-1)

                dev_1= np.mean(dev_matrix_client_1[filtered_indices_list])
                dev_1_v= np.mean(np.abs(set_1_grads - user_grad_org ))
                dev_2= np.mean(dev_matrix_client_1[filtered_indices_list_other])
                dev_2_v = np.mean(np.abs(set_2_grads - user_grad_org ))
                kurt_1 =  np.mean(kurt_all[filtered_indices_list])
                kurt_2 = np.mean(kurt_all[filtered_indices_list_other])

                if np.sum(score_matrix_client[filtered_indices_list_other]) >= np.sum(score_matrix_client[filtered_indices_list]) :#and np.mean(dev_matrix_client[filtered_indices_list_other]) <  np.mean(dev_matrix_client[filtered_indices_list]) :
                  if len(filtered_indices_list_other) !=0  and  (
                       ( grad_sim_v_2 > grad_sim_v_1 and kurt_2 > kurt_1 and dev_2_v < dev_1_v) or 
                       (dev_2_v < dev_1_v and grad_sim_v_2 > grad_sim_v_1 and kurt_2 > kurt_1) or 
                       (grad_sim_v_2 < grad_sim_v_1 and dev_2_v < dev_1_v) or 
                       ( grad_sim_v_2 > grad_sim_v_1 
                        and dev_2_v < dev_1_v and kurt_2 > kurt_1)
                    ):
               
                        filtered_indices_list   = filtered_indices_list_other
             

                 #_______________________________End of Norm Bounding Using MAD__________________________________________
             
                features_list =[]
                layer_grad = []
                list_layer_grad  = []
                deviations  = np.zeros([50, 1])
                #print(filtered_indices_list)

                user_grad_org_all_filtered = [grads_original_copy[i] for i in filtered_indices_list]


                for i in range(len(user_grad_org_all_filtered)):
                    clinet_index = filtered_indices_list[i]
                    client_grad_i= user_grad_org_all_filtered[i].cpu().numpy().flatten()
                    
                    

                    client_grad_i[np.isnan(client_grad_i)] = 0
                    list_layer_grad.append(user_grad_org_all_filtered[i].flatten())
                    vector = client_grad_i
                    

                    layer_grad.append(list(client_grad_i))

                    positive_count = np.sum(vector > 0) 
                    negative_count = np.sum(vector  < 0)
                    zero_count = np.sum(vector == 0)

                    # Basic Statistical Features
                    median_value = np.median(vector)
                    skewness = skew(vector)
                    kurt = kurtosis(vector)
                    
                    # Additional Features
                   
                    norm_v = np.linalg.norm(vector) 

                    absolute_deviation = np.mean(np.abs(vector - user_grad_org_test_layer.detach().cpu().numpy()))
                    user_grad_org_test_layer[np.isnan(client_grad_i)] = 0
                    dir = np.arccos(np.dot(vector,  user_grad_org_test_layer.detach().cpu().numpy()) / (np.linalg.norm(vector) * np.linalg.norm( user_grad_org_test_layer.detach().cpu().numpy())))
                    deviations[clinet_index,0] = deviations[clinet_index,0] + (absolute_deviation)
                                    
                    vector_features = [
                        positive_count, negative_count, zero_count,kurt,skewness
                               ,absolute_deviation,norm_v
                    
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
                    # Create Agglomerative Clustering model
                    agg_clustering = AgglomerativeClustering()

                    # Fit the model and predict cluster labels
                    cluster_labels = agg_clustering.fit_predict(reduced_features)

                    value_indices = {value: [index for index, val in enumerate(cluster_labels) if val == value] for value in set(cluster_labels)}


                    cluster_1_indices = [filtered_indices_list[i]  for i in  value_indices[0]]  
                    cluster_2_indices = [ filtered_indices_list[i]     for i in  value_indices[1]]  
            
                score_cluster_1 = np.sum(score_matrix_client[cluster_1_indices])
                score_cluster_2 = np.sum(score_matrix_client[cluster_2_indices])


                absolute_deviation_1 = np.mean(grads_sim[cluster_1_indices])
                absolute_deviation_2 = np.mean(grads_sim[cluster_2_indices])


                set_1_grads = grads_stacked[cluster_1_indices].mean(dim=0)
                set_2_grads = grads_stacked[cluster_2_indices].mean(dim=0)
                
                set_1_grads = torch.nan_to_num(set_1_grads.detach(), nan=0.0).cpu().numpy().flatten().reshape(1,-1)
                set_2_grads = torch.nan_to_num(set_2_grads.detach(), nan=0.0).cpu().numpy().flatten().reshape(1,-1)
                user_grad_org = torch.nan_to_num(user_grad_org_test_layer.detach(), nan=0.0).cpu().numpy().flatten().reshape(1,-1)

                dev_1= np.mean(np.abs(set_1_grads - user_grad_org ))
                dev_2= np.mean(np.abs(set_2_grads - user_grad_org ))

                kurt_1 =  np.mean(kurt_all[cluster_1_indices])
                kurt_2 = np.mean(kurt_all[cluster_2_indices])

                filtered_indices_list_benign = cluster_1_indices
                
               
                if  len(cluster_2_indices) !=0 and score_cluster_2  >= score_cluster_1 :
                     if (
                       ( absolute_deviation_2 > absolute_deviation_1 and kurt_2 > kurt_1 and dev_2 < dev_1) or 
                       (dev_2 < dev_1 and absolute_deviation_2 > absolute_deviation_1  and kurt_2 > kurt_1) or 
                       (absolute_deviation_2 < absolute_deviation_1 and dev_2 < dev_1) or  ( absolute_deviation_2 > absolute_deviation_1
                        and dev_2 < dev_1 and kurt_2 > kurt_1)
                       ):
                          
                               
                               filtered_indices_list_benign = cluster_2_indices
                               
                if len(cluster_1_indices) ==0 :
                     filtered_indices_list_benign = cluster_2_indices

                benign_indices =benign_indices + filtered_indices_list_benign
                normalized_deviation = 1- (1 / (1 + deviations))
        
                dev_list = np.array(normalized_deviation.tolist())
                score_matrix_client [:,0] +=dev_list[:,0]
                
                list_benign_indices.append(list(filtered_indices_list_benign))
                   

    
            benign_list = list(set(benign_indices))

            global_grad = grads_original
            selected_grads =  global_grad[benign_list,:]
         

            score_matrix_client[benign_list] +=  1
     

            grad_norm = torch.norm(selected_grads, dim=1).reshape((-1, 1))
            norm_clip = grad_norm.median(dim=0)[0].item()
            grad_norm_clipped = torch.clamp(grad_norm,max=norm_clip, out=None)
            grads_clip = (selected_grads/grad_norm)*grad_norm_clipped
            
            global_grad = grads_clip.mean(dim=0)
          
            return global_grad,benign_list,0
         
       
              
