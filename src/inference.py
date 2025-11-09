# ---------------------------------------------------------
# PlantTraitNet Inference Script
# Supports single checkpoint or multiple checkpoints in a directory
# ---------------------------------------------------------

import argparse
import os
import numpy as np
import torch
import time
import joblib
import pandas as pd

from utils import (
    get_config,
    get_logger,
    load_checkpoint,
    set_random_seed,
    plothexbin,
)
from datasets import build_inference_loader
from models import build_model
from torch.backends import cudnn
from timm.utils import AverageMeter


logger = get_logger()


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Evaluation Pipeline")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pth) or directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store inference results")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment naming")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization")
    parser.add_argument("--path_imgref_test", type=str,
                        help="Path to test image reference file if overwriting the config file")
    parser.add_argument("--trait", type=str, default=None, help="Trait to evaluate")
    parser.add_argument("--opts", default=None, nargs="+",
                        help="Modify config options by adding 'KEY VALUE' pairs.")
    return parser.parse_args()


# ---------------------------------------------------------
# Inference for One or Multiple Checkpoints
# ---------------------------------------------------------
def inference(cfg, output_dir, checkpoints, args):
    logger = get_logger()
    transformer_load_path = cfg.data.target_transformer.transformer_path
    target_transformer = joblib.load(transformer_load_path)

    logger.debug(f"Model: {cfg.model.type}")

    data_loader = build_inference_loader(cfg.data, cfg.model.modality, target_transformer=target_transformer)
    dataset = data_loader.dataset

    logger.info(f"Evaluating dataset: {dataset}")
    logger.info(f"Creating model: {cfg.model.type}/{cfg.model_name}")
    model = build_model(cfg.model).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")

    for checkpoint in checkpoints:
        logger.info(f"Inference for checkpoint: {checkpoint}")
        args.resume = checkpoint
        args.eval = True
        cfg = get_config(args)

        suffix = os.path.splitext(os.path.basename(checkpoint))[0]
        tag_suffix = f"{args.tag}_{suffix}" if args.tag else suffix

        ckpt_output_dir = os.path.join(output_dir, f"results_{suffix}")
        os.makedirs(ckpt_output_dir, exist_ok=True)

        max_metrics = load_checkpoint(cfg, model, None, None)
        logger.info(f"Max metrics: {max_metrics}")

        f, normf = validate(
            data_loader,
            target_transformer,
            model=model,
            columns=cfg.data.target_prefix,
            output_path=ckpt_output_dir,
            tag_suffix=tag_suffix,
        )

        result_analysis(f, cfg.data.path_imgref_val, output=ckpt_output_dir,
                        suffix=tag_suffix, trait=args.trait)
        result_analysis(normf, cfg.data.path_imgref_val, output=ckpt_output_dir,
                        suffix=f"norm_{tag_suffix}", trait=args.trait)


# ---------------------------------------------------------
# Result Analysis
# ---------------------------------------------------------
def result_analysis(result, testfile, output, suffix, trait=None):
    data = pd.read_csv(result)
    testfile = pd.read_csv(testfile)
    data = pd.concat([data, testfile], axis=1)

    true = data.columns[data.columns.str.endswith("_true")]
    pred = data.columns[data.columns.str.endswith("_pred")]

    if trait is not None:
        true = [t for t in true if trait in t]
        pred = [p for p in pred if trait in p]

    assert len(true) == len(pred)

    for t, p in zip(true, pred):
        assert t[:-5] == p[:-5]

    trait_name_dict = {
        "SLA": "Specific Leaf Area",
        "SLA_median": "Specific Leaf Area",
        "Height": "Height",
        "Height_median": "Height",
        "Leaf_N": "Leaf Nitrogen",
        "Leaf_N_median": "Leaf Nitrogen",
        "LeafN_median": "Leaf Nitrogen",
        "LeafN": "Leaf Nitrogen",
        "LeafArea": "Leaf Area",
        "LeafArea_median": "Leaf Area",
    }

    for t, p in zip(true, pred):
        print(t, p)
        label = trait_name_dict.get(t.replace("_true", "").replace("_pred", ""), t)
        plothexbin(data, true=t, pred=p, label=label,
                   output_dir=output, suffix=suffix, allmetrics=True, scaleaxis=True)

        for group in data["PFT"].unique():
            print("group:", group)
            group_data = data[data["PFT"] == group]
            plothexbin(group_data, true=t, pred=p, label=label,
                       output_dir=output, suffix=f"{group}_{suffix}", allmetrics=True, scaleaxis=True)


# ---------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------
@torch.no_grad()
def validate(data_loader, target_transformer, model, columns=None, output_path=None, tag_suffix=None):
    logger = get_logger()
    criterion = model.criterion
    model.eval()

    batch_time = AverageMeter()
    start_time = time.time()

    filepath = os.path.join(output_path, f"{tag_suffix}.csv")
    normfilepath = os.path.join(output_path, f"normalized_{tag_suffix}.csv")

    with open(filepath, "w") as f, open(normfilepath, "w") as normf:
        target_col_true = [f"{col}_true" for col in columns]
        target_col_uncertainty = [f"{col}_uncertainty" for col in columns]
        target_col_pred = [f"{col}_pred" for col in columns]

        headers = ",".join(target_col_true + target_col_pred + target_col_uncertainty)
        f.write(headers + "\n")
        normf.write(headers + "\n")

        for idx, samples in enumerate(data_loader):
            logger.info(f"Processing batch {idx}")
            output, log_var = model(**samples, train=False)
            target = samples["label"].cuda()
            loss = criterion(output, log_var, target)

            normalized_samples = np.column_stack(
                (target.cpu().numpy(), output.cpu().numpy(), log_var.cpu().numpy())
            )
            normf.writelines(",".join(map(str, row)) + "\n" for row in normalized_samples)

            if target_transformer:
                inv_target = target_transformer.inverse_transform(target.cpu().numpy())
                inv_output = target_transformer.inverse_transform(output.cpu().numpy())
                inv_samples = np.column_stack((inv_target, inv_output))
                inv_samples = inv_samples[~np.isnan(inv_samples).any(axis=1)]
                f.writelines(",".join(map(str, row)) + "\n" for row in inv_samples)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

    return filepath, normfilepath


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    cudnn.benchmark = True
    config = get_config(args)
    set_random_seed(config.seed)

    # Detect single checkpoint or directory
    if os.path.isdir(args.checkpoint):
        checkpoints = [
            os.path.join(args.checkpoint, f)
            for f in sorted(os.listdir(args.checkpoint))
            if f.endswith(".pth")
        ]
    elif args.checkpoint.endswith(".pth"):
        checkpoints = [args.checkpoint]
    else:
        raise ValueError(f"Invalid checkpoint path: {args.checkpoint}")

    logger.info(f"Checkpoints to evaluate: {checkpoints}")
    inference(config, args.output_dir, checkpoints, args)
