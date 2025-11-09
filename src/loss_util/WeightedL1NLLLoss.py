import torch.nn as nn
import torch
'''
based on https://github.com/EPFL-VILAB/XDEnsembles/blob/master/task_configs.py#L126
'''

class WeightedL1NLLLoss(nn.Module):
    def __init__(self):
        super(WeightedL1NLLLoss, self).__init__()
    
    def forward(self, mu, log_var, targets):  
        log_sigma = torch.clamp(log_var, min=-5, max=5)
        sigma = log_sigma.exp() + 1e-6
        
        lap_dist = torch.distributions.Laplace(loc=mu, scale=sigma)
        logprobs = lap_dist.log_prob(targets)

        loss = -logprobs 
            
        loss = torch.mean(loss, dim=0) #mean over batch
        return loss