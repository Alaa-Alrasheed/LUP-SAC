# coding: utf-8

import tools
import math
import torch
import numpy as np
from typing import List, Optional

# ---------------------------------------------------------------------------- #
class DnC(object):
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.

    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(
        self, num_byzantine=10, *, sub_dim=10000, num_iters=5, filter_frac=1.0
    ) -> None:
        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac

    def aggregate(self, inputs: List[torch.Tensor], f=10, epoch=1, g0=None, iteration=1, **kwargs):  
        self.num_byzantine  = f
        updates = torch.stack(inputs, dim=0)
        d = len(updates[0])

        benign_ids = []
        for i in range(self.num_iters):
            indices = torch.randperm(d)[: self.sub_dim]
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array(
                [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
            )

            good = s.argsort()[
                : len(updates) - int(self.fliter_frac * self.num_byzantine)
            ]
            benign_ids.extend(good)

        benign_ids = list(set(benign_ids))
        benign_updates = updates[benign_ids, :].mean(dim=0)
        return benign_updates,benign_ids,self.num_byzantine

