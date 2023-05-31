import os

import torch
# 根据情况选择模型的device
def select_device(device=''):
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    if not (cpu or mps) and torch.cuda.is_available():
        arg = "cuda:" + device
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        arg = 'mps'
    else:  # revert to CPU
        arg = 'cpu'
    return torch.device(arg)