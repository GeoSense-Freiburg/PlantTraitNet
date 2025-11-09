import torch
import torch.nn as nn

class MixedNLLLoss(nn.Module):
    def __init__(self):
        super(MixedNLLLoss, self).__init__()

    def forward(self, mu, log_var, targets):
        """
        Assumes mu, log_var, targets are of shape (batch_size, num_tasks)
        Uses Laplace for tasks 0 and 1(height and leaf area), Gaussian for tasks 2 and 3(sla and leaf_nitrogen)
        """
        num_tasks = mu.shape[1]
        losses = []

        for i in range(num_tasks):
            task_mu = mu[:, i]
            task_log_var = log_var[:, i]
            task_target = targets[:, i]

            task_log_var = torch.clamp(task_log_var, min=-5, max=5)

            #if i < 2:  # Use Laplace for tasks 0 and 1
            if i == 1:  # Use Laplace only for task 1 (leaf area)
                sigma = task_log_var.exp() + 1e-6
                laplace = torch.distributions.Laplace(loc=task_mu, scale=sigma)
                task_loss = -laplace.log_prob(task_target)
            else:  # Use Gaussian for tasks 2 and 3
                var = task_log_var.exp() + 1e-6
                std = torch.sqrt(var)
                normal = torch.distributions.Normal(loc=task_mu, scale=std)
                task_loss = -normal.log_prob(task_target)

            task_loss = task_loss.mean()  # mean over batch
            losses.append(task_loss)

        return torch.stack(losses)  # shape: (num_tasks,)
