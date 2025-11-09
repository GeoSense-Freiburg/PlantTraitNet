import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from matplotlib.colors import Normalize
from scipy.stats import pearsonr


def nmae_quantile(y_true, y_pred, is_quantile=False, lower_quantile=0.01, upper_quantile=0.99):
    """
    Calculate Normalized Mean Absolute Error (NMAE), optionally normalized by quantile range.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)

    if not is_quantile:
        return mae

    q_low = np.quantile(y_true, lower_quantile)
    q_high = np.quantile(y_true, upper_quantile)
    quantile_range = q_high - q_low

    if quantile_range == 0:
        raise ValueError("Quantile range is zero. Check your data distribution.")

    return mae / quantile_range


def nrmse_quantile(y_true, y_pred, is_quantile=False, lower_quantile=0.01, upper_quantile=0.99):
    """
    Calculate Normalized Root Mean Square Error (NRMSE), optionally normalized by quantile range.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)

    if not is_quantile:
        return rmse

    q_low = np.quantile(y_true, lower_quantile)
    q_high = np.quantile(y_true, upper_quantile)
    quantile_range = q_high - q_low

    if quantile_range == 0:
        raise ValueError("Quantile range is zero. Check your data distribution.")

    return rmse / quantile_range


def pearson_r(y_true, y_pred):
    """Compute the Pearson correlation coefficient."""
    r, _ = pearsonr(y_true, y_pred)
    return r.round(4)


def plothexbin(
    data,
    true,
    pred,
    label,
    scaleaxis=True,
    allmetrics=True,
    norm="range",
    output_dir=None,
    suffix=None,
    log_scale=False,
):
    """Generate a hexbin plot comparing observed vs predicted values."""
    x = data[true]
    y = data[pred]

    print(f"x: {x.shape}, y: {y.shape}")
    print(f"x: {x}, y: {y}")

    if x.shape == (0,) or y.shape == (0,):
        print(f"x: {x.shape}, y: {y.shape}")
        return

    if log_scale:
        true = np.log10(true)
        pred = np.log10(pred)
        label = f"log({label})" if label else "log(values)"

    r2 = r2_score(x, y)

    if allmetrics:
        if norm == "quantile":
            nmae = nmae_quantile(x, y, is_quantile=True)
            nrmse = nrmse_quantile(x, y, is_quantile=True)
        elif norm == "mean":
            nmae = mean_absolute_error(x, y) / x.mean()
            nrmse = root_mean_squared_error(x, y) / x.mean()
        elif norm == "range":
            nmae = mean_absolute_error(x, y) / (x.max() - x.min())
            nrmse = root_mean_squared_error(x, y) / (x.max() - x.min())

        pearson_r_value = pearson_r(x, y)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"Observed {label}", fontsize=20, fontweight="bold")
    ax.set_ylabel(f"Predicted {label}", fontsize=20, fontweight="bold")

    if scaleaxis:
        maxval = max(x.max(), y.max())
        minval = min(x.min(), y.min())
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
        ax.plot([minval, maxval], [minval, maxval], color="red", lw=2)
        extent = [minval, maxval, minval, maxval]
    else:
        extent = None

    hb = ax.hexbin(
        x,
        y,
        gridsize=60,
        cmap="plasma",
        mincnt=1,
        extent=extent,
        norm=Normalize(vmin=0, vmax=1),
    )

    counts = hb.get_array()
    norm_counts = counts / counts.max()
    hb.set_array(norm_counts)

    if allmetrics:
        ax.text(
            0.05,
            0.95,
            f"R²: {r2:.2f}\nnMAE: {nmae:.2f}\nnRMSE: {nrmse:.2f}\nr: {pearson_r_value:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
        )
    else:
        ax.text(
            0.05,
            0.95,
            f"R²: {r2:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
        )

    print(f"{label} R²: {r2:.2f}")
    if allmetrics:
        print(f"{label} nMAE: {nmae:.2f}")
        print(f"{label} nRMSE: {nrmse:.2f}")

    plt.tight_layout(pad=0.5)

    path = f"{label}.png"
    if output_dir is not None:
        path = os.path.join(output_dir, path)
    if suffix is not None:
        path = path.replace(".png", f"_{suffix}.png")

    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
