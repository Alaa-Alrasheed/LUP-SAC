
import math
import torch
import numpy as np

def MPAF(byz_grads, *args, **kwargs):
    mp_lambda = 10.0
    num_byzs = len(byz_grads)
    fake_grads = []
    if num_byzs == 0:
        return list()
    for byz_grad in byz_grads:
         tmp = torch.zeros_like(byz_grad, dtype = torch.float32)
         w_base = torch.randn_like(byz_grad, dtype = torch.float32)
         tmp += (byz_grad- w_base) *   mp_lambda 
         fake_grads.append(tmp)
  
    return fake_grads


