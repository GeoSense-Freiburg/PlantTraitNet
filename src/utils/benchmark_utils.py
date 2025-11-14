"""
Utility functions for benchmarking.
"""
import os
import warnings
import pandas as pd
from typing import Iterable, Union, Optional, List, Dict
from pyproj import Proj
from shapely.geometry import shape
import json
from pathlib import Path
from box import ConfigBox
import xarray as xr
import rioxarray as riox

from utils.benchmark_conf import get_benchmark_config

cfg = get_benchmark_config()


def global_grid_df(
    df: pd.DataFrame,
    col: str,
    lon: str = "decimallongitude",
    lat: str = "decimallatitude",
    res: Union[int, float] = 0.5,
    stats: Optional[List[str]] = None,
    n_min: int = 1,
) -> pd.DataFrame:
    """
    Calculate gridded statistics for a given DataFrame.
    """
    warnings.warn(
        "'global_grid_df' is deprecated and will be removed in a future "
        "version. Use 'rasterize_points' instead.",
        DeprecationWarning,
    )

    stat_funcs = {
        "mean": "mean",
    }

    df = df.copy()
    df["y"] = (df[lat] + 90) // res * res - 90 + res / 2
    df["x"] = (df[lon] + 180) // res * res - 180 + res / 2

    gridded_df = (
        df.drop(columns=[lat, lon])
        .groupby(["y", "x"], observed=False)[[col]]
        .agg(list(stat_funcs.values()))
    )
    #gridded df heaf
    print(gridded_df.head())

    gridded_df.columns = list(stat_funcs.keys())
    print(gridded_df.head())

    if n_min > 1:
        gridded_df = gridded_df[gridded_df["count"] >= n_min]

    if stats is not None:
        return gridded_df[stats]

    return gridded_df


def get_lat_area(lat: Union[int, float], resolution: Union[int, float]) -> float:
    """Calculate the area of a grid cell at a given latitude."""
    coordinates = [
        (0, lat + (resolution / 2)),
        (resolution, lat + (resolution / 2)),
        (resolution, lat - (resolution / 2)),
        (0, lat - (resolution / 2)),
        (0, lat + (resolution / 2)),
    ]

    projection_string = (
        f"+proj=aea +lat_1={coordinates[0][1]} +lat_2={coordinates[2][1]} "
        f"+lat_0={lat} +lon_0={resolution / 2}"
    )
    pa = Proj(projection_string)

    x, y = pa(*zip(*coordinates))
    area = shape({"type": "Polygon", "coordinates": [list(zip(x, y))]}).area / 1e6

    return area


def lat_weights(lat_unique: Iterable[Union[int, float]], resolution: Union[int, float]) -> Dict[Union[int, float], float]:
    """Calculate weights for each latitude band based on area of grid cells."""
    weights = {}
    for j in lat_unique:
        weights[j] = get_lat_area(j, resolution)

    max_area = max(weights.values())
    weights = {k: v / max_area for k, v in weights.items()}

    return weights


def open_raster(
    filename: Union[str, os.PathLike], mask_and_scale: bool = True, **kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    """Open a raster dataset using rioxarray."""
    ds = riox.open_rasterio(filename, mask_and_scale=mask_and_scale, **kwargs)
    if isinstance(ds, list):
        raise ValueError("Multiple files found.")
    return ds


def check_y_set(y_set: str) -> None:
    """Check if the specified y_set is valid."""
    y_sets = ["splot"]
    if y_set not in y_sets:
        raise ValueError(f"Invalid y_set. Must be one of {y_sets}.")


def get_trait_maps_dir(y_set: str, config: ConfigBox = cfg) -> Path:
    """Get the path to the trait maps directory for a specific dataset (e.g. GBIF or sPlot)."""
    check_y_set(y_set)
    return (
        Path(config.interim_dir)
    )


def read_trait_map(
    trait_id: str, y_set: str, config: ConfigBox = cfg, band: Optional[int] = None
) -> Union[xr.DataArray, xr.Dataset]:
    """Read and return a specific trait map as an xarray DataArray or Dataset."""
    check_y_set(y_set)
    fn = get_trait_maps_dir(y_set, config) / f"{trait_id}.tif"
    if band is not None:
        return open_raster(fn).sel(band=band)
    return open_raster(fn)
