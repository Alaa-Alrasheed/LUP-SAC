from typing import List

import torch
import numpy as np
from numpy import inf
from sklearn.cluster import AgglomerativeClustering, KMeans
from typing import Iterable, Dict, Optional, TYPE_CHECKING, Union


_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def clip_tensor_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.detach(), norm_type).to(device)
                    for p in parameters
                    if p.dtype != torch.int64
                ]
            ),
            norm_type,
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1,
    # but doing so avoids a `if clip_coef < 1:` conditional which can require a
    # CPU <=> device synchronization when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.dtype != torch.int64:
            return p.detach().mul_(clip_coef_clamped.to(p.device))
        
def _mean(inputs: List[torch.Tensor]):
    inputs_tensor = torch.stack(inputs, dim=0)
    return inputs_tensor.mean(dim=0)


def _median(inputs: List[torch.Tensor]):
    inputs_tensor = torch.stack(inputs, dim=0)
    values_upper, _ = inputs_tensor.median(dim=0)
    values_lower, _ = (-inputs_tensor).median(dim=0)
    return (values_upper - values_lower) / 2


class Mean(object):
    def __call__(self, inputs: List[torch.Tensor]):
        return _mean(inputs)


        
class Median(object):
    def __call__(self, inputs: List[torch.Tensor]):
        return _median(inputs)
class Clippedclustering(object):
    def __init__(
        self, agg="mean", signguard=False, max_tau=1e5, linkage="average"
       
    ) -> None:
        
        super(Clippedclustering, self).__init__()

        assert linkage in ["average", "single"]
        self.tau = max_tau
        self.signguard = signguard
        self.linkage = linkage
        self.l2norm_his = []
        self.name = "Clippedclustering"
        if agg == "mean":
            self.agg = Mean()
        elif agg == "median":
            self.agg = Median()
        else:
            raise NotImplementedError(f"{agg} is not supported yet.")

    def aggregate(self, inputs: List[torch.Tensor], f=10, epoch=1, g0=None, iteration=1, **kwargs):  
        updates = torch.stack(inputs, dim=0)
        l2norms = [torch.norm(update).item() for update in updates]
        self.l2norm_his.extend(l2norms)
        threshold = np.median(self.l2norm_his)
        threshold = min(threshold, self.tau)

        for idx, l2 in enumerate(l2norms):
            if l2 > threshold:
                updates[idx] = clip_tensor_norm_(updates[idx], threshold)

        num = len(updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                    updates[i, :], updates[j, :], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -inf] = 0
        dis_max[dis_max == inf] = 2
        dis_max[np.isnan(dis_max)] = 2
        clustering = AgglomerativeClustering(
             linkage=self.linkage, n_clusters=2
        )
        clustering.fit(dis_max)

        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        S1_idxs = list(
            [idx for idx, label in enumerate(clustering.labels_) if label == flag]
        )
        selected_idxs = S1_idxs

        if self.signguard:
            features = []
            num_para = len(updates[0])
            for update in updates:
                feature0 = (update > 0).sum().item() / num_para
                feature1 = (update < 0).sum().item() / num_para
                feature2 = (update == 0).sum().item() / num_para

                features.append([feature0, feature1, feature2])

            kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

            flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
            S2_idxs = list(
                [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
            )

            selected_idxs = list(set(S1_idxs) & set(S2_idxs))

        benign_updates = []
        for idx in selected_idxs:
            benign_updates.append(updates[idx])

        values = self.agg(benign_updates)
        return values,selected_idxs,0