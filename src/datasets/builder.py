# -------------------------------------------------------------------------
# Dataloader Builder (Single GPU Version)
# -------------------------------------------------------------------------
import warnings
import torch
from torchvision import transforms
import pandas as pd
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import parse_dataset, parse_inf_dataset, get_logger
from .dataset import Dataset
from torch.utils.data import WeightedRandomSampler, RandomSampler, SequentialSampler

# -------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------
def warn_and_continue(exn):
    """Ignore exceptions, issue a warning, and continue."""
    warnings.warn(repr(exn))
    return True


# -------------------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate function for image/geo datasets.
    Ensures proper stacking and error handling.
    """
    if len(batch) == 0:
        raise ValueError("All samples in the batch are None. Check dataset loading.")

    out = {}
    for key in batch[0].keys():
        try:
            out[key] = torch.stack([b[key] for b in batch])
        except Exception as e:
            raise RuntimeError(f"Error stacking key '{key}': {e}")
    return out


# -------------------------------------------------------------------------
# Image Transform Builder
# -------------------------------------------------------------------------
def build_img_transform(dataset, config=None):
    """Builds standard image transforms for train/test."""
    if dataset == 'train':
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


# -------------------------------------------------------------------------
# Dataset Builder
# -------------------------------------------------------------------------
def build_dataset(
    dataframe,
    config,
    target_transformer,
    is_img_transform=True,
    col=None,
    modality='image_geo',
    dataset='train'
):
    """Creates dataset for train/val/test phases."""
    img_transform = build_img_transform(dataset=dataset) if is_img_transform else None

    dataset_obj = Dataset(
        dataframe=dataframe,
        img_transform=img_transform,
        columns=col,
        modality=modality,
        phase=dataset,
        base_seed=config.seed if dataset == 'train' else None,
        target_transformer=target_transformer,
        data_path=config.data_path
    )

    print(f"Dataset built ({dataset}): {len(dataset_obj)} samples")
    return dataset_obj


# -------------------------------------------------------------------------
# Dataloader Builder
# -------------------------------------------------------------------------
def build_loader(config, modality='image_geo', target_transformer=None, logger=None):
    """Build train/val/test dataloaders for multimodal datasets."""
    if logger is None:
        logger = get_logger()

    #train_dataframes, val_dataframes, test_dataframes = [], [], []
    train_columns, val_columns = None, None

    # Parse dataset splits
    train_dat, val_dat, train_columns, val_columns = parse_dataset(config)


    logger.info(f"Frames - Train: {len(train_dat)}, Val: {len(val_dat)}")

    # train_dat = pd.concat(train_dataframes)
    # val_dat = pd.concat(val_dataframes)

    dataset_train = build_dataset(
        dataset='train',
        dataframe=train_dat,
        config=config,
        col=train_columns,
        modality=modality,
        target_transformer=target_transformer, 
    )
    logger.info(f"Train dataset built: {len(dataset_train)}")

    dataset_val = build_dataset(
        dataset='val',
        dataframe=val_dat,
        config=config,
        col=val_columns,
        modality=modality,
        target_transformer=target_transformer,
    )
    logger.info(f"Val dataset built: {len(dataset_val)}")

    # Weighted or random sampler
    if getattr(config, "weighted_sampling", False):
        species_count = dataset_train.df[config.weighted_col].value_counts()
        logger.info(f"Species count: {species_count}")
        weights = 1.0 / species_count
        samples_weight = weights[dataset_train.df[config.weighted_col]].tolist()
        logger.info(f"Species count: {species_count}")

        sampler_train = WeightedRandomSampler(
            weights=torch.DoubleTensor(samples_weight),
            num_samples=len(samples_weight),
            replacement=True
    )
    else:
        logger.info("Weighted sampling disabled.")
        sampler_train = RandomSampler(dataset_train)

    sampler_val = SequentialSampler(dataset_val)

    # DataLoaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.batch_size,
        num_workers=config.train_dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.batch_size,
        num_workers=config.val_dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    logger.info(f"len(dataloader_train): {len(data_loader_train)}")
    logger.info(f"len(dataloader_val): {len(data_loader_val)}")

    return dataset_train, data_loader_train, data_loader_val


# -------------------------------------------------------------------------
# Inference Loader
# -------------------------------------------------------------------------
def build_inference_loader(config, modality='multimodal', target_transformer=None):
    """Build dataloader for inference/evaluation."""
    test_dataframes = []
    #for train in config.train:
        
    test_dat, columns = parse_inf_dataset(config, target_transformer=target_transformer)
    test_dataframes.append(test_dat)

    test_dat = pd.concat(test_dataframes)

    dataset_test = build_dataset(
        dataset='test',
        dataframe=test_dat,
        config=config,
        col=columns,
        modality=modality,
        target_transformer=target_transformer,
    )

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=config.batch_size,
        num_workers=config.test_dataloader_workers if hasattr(config, 'test_dataloader_workers') else config.val_dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return data_loader_test
