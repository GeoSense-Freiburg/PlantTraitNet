import os
from collections import defaultdict

import torch
from mmcv.runner import CheckpointLoader
from omegaconf import read_write
from .logger import get_logger

# def load_checkpoint(config, model, optimizer, lr_scheduler):
#     """
#     Load full training checkpoint.

#     Returns:
#         metrics (dict): Restored metrics dictionary.
#     """
#     logger = get_logger()
#     logger.info(f'==============> Resuming from {config.checkpoint.resume} ...')

#     # --- FIX for PyTorch 2.6+ safe unpickling ---
#     try:
#         import omegaconf
#         torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
#     except Exception as e:
#         logger.warning(f"Safe globals registration failed: {e}")
#     # --------------------------------------------

#     # --- Compatible load for MMCV vs PyTorch 2.6 ---
#     try:
#         checkpoint = CheckpointLoader.load_checkpoint(
#             config.checkpoint.resume, map_location='cpu', weights_only=False
#         )
#     except TypeError:
#         # Older MMCV versions don't support `weights_only`
#         logger.warning("[WARN] MMCV load_checkpoint() does not support `weights_only`. Using torch.load() instead.")
#         checkpoint = torch.load(config.checkpoint.resume, map_location='cpu')
#     # -----------------------------------------------

#     msg = model.load_state_dict(checkpoint['model'], strict=False)
#     logger.info(msg)

#     metrics = defaultdict(float)
#     if (not config.evaluate.eval_only and not config.checkpoint.loadonlymodel
#             and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint
#             and 'epoch' in checkpoint):
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         with read_write(config):
#             config.train.start_epoch = checkpoint['epoch'] + 1
#         logger.info(f"=> loaded successfully '{config.checkpoint.resume}' (epoch {checkpoint['epoch']})")
#         metrics = checkpoint.get('metrics', metrics)

#     torch.cuda.empty_cache()
#     return metrics
def load_checkpoint(config, model, optimizer, lr_scheduler):
    """
    Load full training checkpoint.

    Returns:
        metrics (dict): Restored metrics dictionary.
    """
    logger = get_logger()
    logger.info(f'==============> Resuming from {config.checkpoint.resume} ...')

    # --- FIX for PyTorch 2.6+ safe unpickling ---
    try:
        import omegaconf
        torch.serialization.add_safe_globals([
            omegaconf.listconfig.ListConfig,
            omegaconf.base.ContainerMetadata
        ])
        logger.info("Registered omegaconf safe globals for torch.load.")
    except Exception as e:
        logger.warning(f"Safe globals registration failed: {e}")
    # --------------------------------------------

    # --- Compatible MMCV vs plain torch.load ---
    try:
        checkpoint = CheckpointLoader.load_checkpoint(
            config.checkpoint.resume, map_location='cpu', weights_only=False
        )
    except TypeError:
        # MMCV version without `weights_only`
        logger.warning("[WARN] MMCV load_checkpoint() does not support `weights_only`. Using torch.load() instead.")
        # Explicitly set weights_only=False for PyTorch 2.6+
        checkpoint = torch.load(config.checkpoint.resume, map_location='cpu', weights_only=False)
    # --------------------------------------------

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    metrics = defaultdict(float)
    if (not config.evaluate.eval_only and not config.checkpoint.loadonlymodel
            and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        with read_write(config):
            config.train.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"=> loaded successfully '{config.checkpoint.resume}' (epoch {checkpoint['epoch']})")
        metrics = checkpoint.get('metrics', metrics)

    torch.cuda.empty_cache()
    return metrics


# def load_checkpoint(config, model, optimizer, lr_scheduler):
#     """
#     Load full training checkpoint.

#     Returns:
#         metrics (dict): Restored metrics dictionary.
#     """
#     logger = get_logger()
#     logger.info(f'==============> Resuming from {config.checkpoint.resume} ...')

#     # --- FIX for PyTorch 2.6+ safe unpickling ---
#     try:
#         import omegaconf
#         torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
#     except Exception as e:
#         logger.warning(f"Safe globals registration failed: {e}")
#     # --------------------------------------------

#     #Weights only set to False to load optimizer, lr_scheduler states, config and epoch to resume training,
#     #set to True if only model weights are needed or if you dont trust the checkpoint source
#     checkpoint = CheckpointLoader.load_checkpoint(config.checkpoint.resume, map_location='cpu', weights_only=False)
#     msg = model.load_state_dict(checkpoint['model'], strict=False)
#     logger.info(msg)

#     metrics = defaultdict(float)
#     if (not config.evaluate.eval_only and not config.checkpoint.loadonlymodel
#             and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint
#             and 'epoch' in checkpoint):
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         with read_write(config):
#             config.train.start_epoch = checkpoint['epoch'] + 1
#         logger.info(f"=> loaded successfully '{config.checkpoint.resume}' (epoch {checkpoint['epoch']})")
#         metrics = checkpoint.get('metrics', metrics)

#     torch.cuda.empty_cache()
#     return metrics


def save_checkpoint(config, epoch, model, metrics, optimizer, lr_scheduler, suffix=''):
    """
    Save model and optimizer checkpoint.
    """
    logger = get_logger()

    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'metrics': metrics,
        'epoch': epoch,
        'config': config
    }

    for k, v in metrics.items():
        save_state[k] = v

    if len(suffix) > 0 and not suffix.startswith('_'):
        suffix = '_' + suffix
        filename = f'ckpt_epoch_{epoch}{suffix}.pth'
        save_path = os.path.join(config.output, filename)
    else:
        save_path = os.path.join(config.output, 'checkpoint.pth')

    logger.info(f"Saving checkpoint: {save_path}")
    torch.save(save_state, save_path)
    logger.info("Checkpoint saved successfully.")


def get_grad_norm(parameters, norm_type=2):
    """
    Compute gradient norm of parameters.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    """
    Automatically locate the latest checkpoint file in output directory.
    """
    default_ckpt = os.path.join(output_dir, 'checkpoint.pth')
    if os.path.exists(default_ckpt):
        return default_ckpt

    checkpoints = [ckpt for ckpt in os.listdir(output_dir) if ckpt.endswith('.pth')]
    print(f'All checkpoints found in {output_dir}: {checkpoints}')

    if checkpoints:
        latest_ckpt = max(
            (os.path.join(output_dir, ckpt) for ckpt in checkpoints),
            key=os.path.getmtime
        )
        print(f'Latest checkpoint found: {latest_ckpt}')
        return latest_ckpt

    return None
