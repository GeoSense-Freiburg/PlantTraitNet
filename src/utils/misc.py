import torch
from collections import OrderedDict
import random
import numpy as np

class ListAverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val):

        if self.val is None:
            # initialize the self.val list with 0 values, use length of val
            self.val = [0] * len(val)
            self.sum = [0] * len(val)
            self.avg = [0] * len(val)

        print("val:",len(val))
        print("self.val:",len(self.val))
        

        for i in range(len(val)):
            self.count += 1
            self.sum[i] = self.val[i] + val[i] 
            self.avg[i] = self.sum[i] / self.count
        self.val = val


def parse_losses(losses, mean=True):
    '''Parse the losses dict and return the total loss and log_vars.'''
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    if not mean: 
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    else:
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key) / len(log_vars)

    return loss, log_vars

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm

def set_random_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.
    
    Args:
        seed (int): The base seed value.
    """
    # Calculate a unique seed for this process
    unique_seed = seed 

    # Set seed for Python's built-in random module
    random.seed(unique_seed)
    
    # Set seed for NumPy
    np.random.seed(unique_seed)
    
    # Set seed for PyTorch
    torch.manual_seed(unique_seed)
    
    # Ensure deterministic behavior in CUDA (if using GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(unique_seed)
        torch.cuda.manual_seed_all(unique_seed)  # If using multi-GPU

    # Optionally set additional settings for PyTorch (to ensure full determinism)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
