"""
PlantTraitNet Training and Evaluation Script
"""
import argparse
import os
import os.path as osp
import time
import math
import datetime
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.backends import cudnn
from timm.utils import AverageMeter
from sklearn.metrics import r2_score
import sklearn.preprocessing as processing
import joblib
from omegaconf import OmegaConf

from utils import (
    get_config,
    get_logger,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
    auto_resume_helper,
    set_random_seed,
    load_class_from_config,
)
from datasets import build_loader
from models import build_model

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("PANOPS training and evaluation script (single GPU)")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--opts", nargs="+", default=None, help="Modify config options as KEY=VALUE pairs")
    parser.add_argument("--batch-size", type=int, help="Batch size for single GPU")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="./output", help="Root of output folder: <output>/<model_name>/<tag>")
    parser.add_argument("--tag", type=str, help="Tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--keep", type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode")
    return parser.parse_args()


# ---------------------------------------------------------
# Target Transformer
# ---------------------------------------------------------
def get_target_transformer(cfg):
    if cfg.data.target_transformer.preload:
        return joblib.load(cfg.data.target_transformer.transformer_path)
    else:
        return load_class_from_config(cfg.data, "target_transformer", processing)


# ---------------------------------------------------------
# Train for One Epoch
# ---------------------------------------------------------
def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, target_transformer):
    logger = get_logger()
    model.train()
    optimizer.zero_grad()

    wandb = None
    if config.wandb:
        import wandb

    num_steps = len(data_loader)
    batch_time, loss_meter, norm_meter = AverageMeter(), AverageMeter(), AverageMeter()
    all_uncertainty_meter = defaultdict(AverageMeter)
    all_loss_meter = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    for idx, samples in enumerate(data_loader):
        logger.info(f"Iteration: {idx}")
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        losses, log_vars = model(**samples)
        loss = losses["nll"]
        loss.backward()

        grad_norm = None
        if config.train.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)

        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()

        loss_meter.update(loss.item())
        for i in range(log_vars.shape[1]):
            all_uncertainty_meter[f"log_var_{i}"].update(log_vars[i].mean().item())
        for key, value in losses.items():
            all_loss_meter[key].update(value.item())
        if grad_norm:
            norm_meter.update(grad_norm)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            eta = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}] "
                f"eta {datetime.timedelta(seconds=int(eta))} "
                f"lr {lr:.6f} "
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f}) "
                f"total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f}) "
                f"mem {memory_used:.2f} GB"
            )

            if config.wandb:
                log_stat = {"iter/nll_loss": loss_meter.avg, "iter/learning_rate": lr}
                for key, val in all_uncertainty_meter.items():
                    log_stat[f"iter/uncertainty_{key}"] = val.avg
                for key, val in all_loss_meter.items():
                    log_stat[f"iter/loss_{key}"] = val.avg
                wandb.log(log_stat)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training took {datetime.timedelta(seconds=int(epoch_time))}")
    return {"total_loss": loss_meter.avg}


# ---------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------
@torch.no_grad()
def validate(config, data_loader, target_transformer, model):
    columns = config.data.target_prefix
    logger = get_logger()
    model.eval()

    batch_time, loss_meter, r2_meter, inv_r2_meter, uncertainty_meter = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    allr2_meter = defaultdict(AverageMeter)
    inv_allr2_meter = defaultdict(AverageMeter)

    logger.info(f"Validation started. Dataset size: {len(data_loader)}")
    end = time.time()

    for idx, samples in enumerate(data_loader):
        logger.info(f"Validation iteration: {idx}")

        output, log_var = model(**samples, train=False)
        labels = samples["label"].cuda()


        losses = model.criterion(output, log_var, labels)
        output_np = np.nan_to_num(output.cpu().numpy())
        labels_np = labels.cpu().numpy()

        r2 = r2_score(labels_np, output_np)
        r2_scores = r2_score(labels_np, output_np, multioutput="raw_values")
        for i, r2_val in enumerate(r2_scores):
            allr2_meter[columns[i]].update(r2_val)

        inv_target = target_transformer.inverse_transform(labels_np)
        inv_output = target_transformer.inverse_transform(output_np)
        inv_r2 = r2_score(inv_target, inv_output)
        inv_r2_scores = r2_score(inv_target, inv_output, multioutput="raw_values")
        for i, inv_val in enumerate(inv_r2_scores):
            inv_allr2_meter[columns[i]].update(inv_val)

        loss_meter.update(losses.mean().item())
        r2_meter.update(r2)
        inv_r2_meter.update(inv_r2)
        uncertainty_meter.update(log_var.mean().item())
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            mem = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"R2 {r2_meter.val:.3f} ({r2_meter.avg:.3f}) "
                f"invR2 {inv_r2_meter.val:.3f} ({inv_r2_meter.avg:.3f}) "
                f"Mem {mem:.0f} MB"
            )

    logger.info(
        f" * Final R2 {r2_meter.avg:.3f}, invR2 {inv_r2_meter.avg:.3f}, loss {loss_meter.avg:.3f}"
    )
    return r2_meter.avg, inv_r2_meter.avg, loss_meter.avg, allr2_meter, inv_allr2_meter, uncertainty_meter


# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------
def train(cfg):
    logger = get_logger()

    if cfg.wandb:
        import wandb
        wandb.init(
            project="panops2",
            name=osp.join(cfg.model_name, cfg.tag),
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=cfg.checkpoint.auto_resume,
        )

    target_transformer = get_target_transformer(cfg)
    dataset_train, loader_train, loader_val = build_loader(
        cfg.data, cfg.model.modality, target_transformer=target_transformer
    )

    model = build_model(cfg.model).cuda()
    optimizer = build_optimizer(cfg.train, model)
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(loader_train))

    max_metrics = {
        "max_val_r2": -math.inf,
        "inv_max_val_r2": -math.inf,
        "min_val_loss": math.inf,

    }

    if cfg.checkpoint.auto_resume:
        resume_file = auto_resume_helper(cfg.output)
        if resume_file:
            cfg.checkpoint.resume = resume_file
            logger.info(f"Auto-resuming from {resume_file}")
        else:
            logger.info("No checkpoint found for auto-resume")

    if cfg.checkpoint.resume:
        max_metrics = load_checkpoint(cfg, model, optimizer, lr_scheduler)

    logger.info(f"Starting training from epoch {cfg.train.start_epoch}")
    start_time = time.time()

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        torch.cuda.empty_cache()
        logger.info(f"--- Epoch {epoch} starts ---")

        train_loss_dict = train_one_epoch(cfg, model, loader_train, optimizer, epoch, lr_scheduler, target_transformer)
        train_loss = train_loss_dict["total_loss"]
        logger.info(f"Avg training loss: {train_loss:.4f}")

        val_r2, inv_val_r2, val_loss, _, _, _ = validate(cfg, loader_val, target_transformer, model)
        logger.info(
            f"Validation Results - Epoch {epoch}: "
            f"R2={val_r2:.3f}, invR2={inv_val_r2:.3f}, loss={val_loss:.4f}"
        )
        if val_r2 > max_metrics["max_val_r2"]:
            max_metrics["max_val_r2"] = val_r2
            max_metrics["inv_max_val_r2"] = inv_val_r2
            max_metrics["min_val_loss"] = val_loss
            logger.info(f"New max_val_r2: {val_r2:.3f}. Checkpointing...")
            save_checkpoint(
                cfg,
                epoch,
                model,
                max_metrics,
                optimizer,
                lr_scheduler,
                suffix="best_val_r2",
            )
                
        if (epoch % cfg.checkpoint.save_freq == 0) or (epoch == cfg.train.epochs - 1):
            save_checkpoint(
                cfg,
                epoch,
                model,
                max_metrics,
                optimizer,
                lr_scheduler,
                suffix=f"train_epoch_{epoch}",
            )

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training completed in {total_time}")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    config = get_config(args)
    logger = get_logger(config)

    set_random_seed(config.seed)
    cudnn.benchmark = True

    os.makedirs(config.output, exist_ok=True)
    train(config)
