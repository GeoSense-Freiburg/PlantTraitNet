from torch import optim
from utils import get_logger


def check_items_in_string(items, key: str) -> bool:
    """Return True if any substring in `items` is found in `key`."""
    return any(item in key for item in items)


def check_keywords_in_name(name: str, keywords=()) -> bool:
    """Return True if any keyword is found in the given name."""
    return any(keyword in name for keyword in keywords)


def set_gradient(model, finetune: dict):
    """
    Freeze selected layers based on provided finetune configuration.

    Args:
        model: The model whose gradients will be set.
        finetune (dict): Mapping of layer groups to substrings to freeze.

    Returns:
        model: The model with gradients appropriately set.
    """
    logger = get_logger()

    n_parameters_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if finetune:
        for _, value in finetune.items():
            for name, param in model.named_parameters():
                if check_items_in_string(value, name):
                    param.requires_grad = False
                    logger.debug(f"Freezing layer: {name}")

    n_parameters_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_parameters_after} (was {n_parameters_before})")

    return model


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """
    Separate parameters into decay and no-decay groups for optimizer construction.

    Args:
        model: Model to process.
        skip_list (tuple): Specific parameters to skip decay.
        skip_keywords (tuple): Keywords identifying params to skip decay.

    Returns:
        list[dict]: Parameter groups with and without weight decay.
    """
    has_decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [
        {"params": has_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(config, model):
    """
    Build optimizer with optional layer freezing and weight decay handling.

    Args:
        config: Configuration object containing optimizer and training parameters.
        model: Model to optimize.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    logger = get_logger()

    # Freeze layers if finetuning is specified
    if config.finetune:
        model = set_gradient(model, config.finetune)

    parameters = set_weight_decay(model)

    opt_name = config.optimizer.name.lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(
            parameters,
            lr=config.optimizer.base_lr,
            eps=config.optimizer.eps,
            betas=config.optimizer.betas,
            weight_decay=config.weight_decay,
        )
        logger.info(
            f"Built AdamW optimizer | lr={config.optimizer.base_lr}, "
            f"eps={config.optimizer.eps}, betas={config.optimizer.betas}, "
            f"weight_decay={config.weight_decay}"
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return optimizer
