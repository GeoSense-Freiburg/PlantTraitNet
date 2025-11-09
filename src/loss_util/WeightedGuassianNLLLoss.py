import torch
import torch.nn as nn

class WeightedGaussianNLLLoss(nn.Module):
    def __init__(self):
        super(WeightedGaussianNLLLoss, self).__init__()

    def forward(self, mu, log_var, targets):
        # Clamp log variance for stability
        log_var = torch.clamp(log_var, min=-5, max=5)
        
        std = (0.5 * log_var).exp() #calculating std

        # Define normal distribution
        normal_dist = torch.distributions.Normal(loc=mu, scale=std)
        logprobs = normal_dist.log_prob(targets)

        # Negative log-likelihood
        loss = -logprobs
        loss = torch.mean(loss, dim=0)  # mean over batch
        return loss
