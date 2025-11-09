import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import io
from scipy.stats import truncnorm
from utils import get_logger
from pathlib import Path  # add this import at the top



def truncated_normal(mean, std, lower, upper, size=1, seed=None, tol=1e-6):
    # Ensure arrays are numpy and shaped (4, 1)
    global_random_state = np.random.get_state()  # Preserve random state
    reshape_dim = lower.shape[0] if isinstance(mean, (list, np.ndarray)) else 1
    
    mean = np.asarray(mean).reshape(reshape_dim, 1)
    std = np.asarray(std).reshape(reshape_dim, 1)
    std = np.nan_to_num(std, nan=1e-6)
    std = np.clip(std, 1e-6, None)
    lower = np.asarray(lower).reshape(reshape_dim, 1)
    upper = np.asarray(upper).reshape(reshape_dim, 1)

    
    # Compute truncated bounds and clip a, b >= 0
    a = np.clip((lower - mean) / std, 1e-4, None)
    b = np.clip((upper - mean) / std, 1e-4, None)
    

    try:
        np.random.seed(seed)
        close_mask = np.isclose(a, b, atol=tol)
        samples = mean.copy()

        # Sample where a ≠ b
        if np.any(~close_mask):
            samples[~close_mask] = truncnorm.rvs(
                a[~close_mask],
                b[~close_mask],
                loc=mean[~close_mask],
                scale=std[~close_mask],
                size=np.count_nonzero(~close_mask)
            )
        return samples
    except Exception as e:
        print(f"exception:{e} for a:{a.ravel()}, b:{b.ravel()}, std:{std.ravel()}, mean:{mean.ravel()}, lower:{lower.ravel()}, upper:{upper.ravel()}")
        return mean
    finally:
        np.random.set_state(global_random_state)  # Restore original state
    


class Dataset(Dataset):
    def __init__(self, dataframe, target_transformer, data_path, img_transform=None, columns=None,
                modality='image', phase='train', base_seed=0,
                geo_location=False):
        """
        Args:
            dataframe (pd.DataFrame): dataset
            target_transformer: label scaler/log-transformer 
            img_transform: torchvision transforms
            columns (dict): contains target and geo column names
            modality (str): 'image' or 'image_geo'
            phase (str): 'train', 'val', or 'test'
            base_seed (int): random seed base
            geo_location (bool): whether to use longitude/latitude
        """
        self.logger = get_logger()
        self.df = dataframe
        self.transform = img_transform
        self.modality = modality
        self.phase = phase
        self.base_seed = base_seed
        self.target_transform = target_transformer
        self.geo_location = geo_location
        self.epoch = 0
        
        self.data_path = data_path

        # Target column indices
        if phase == 'train':
            self.target_lower_idx = [self.df.columns.get_loc(c) for c in columns['target_lowerbound_columns']]
            self.target_upper_idx = [self.df.columns.get_loc(c) for c in columns['target_upperbound_columns']]
            self.target_median_idx = [self.df.columns.get_loc(c) for c in columns['target_median_columns']]
            self.target_std_idx = [self.df.columns.get_loc(c) for c in columns['target_std_columns']]
        else:
            self.target_label_idx = [self.df.columns.get_loc(c) for c in columns['target_label_columns']]

        # Geo columns (only if image_geo or geo_location=True)
        if modality == 'image_geo' or geo_location:
            self.geo_idx = [self.df.columns.get_loc(c) for c in ['Longitude', 'Latitude']]

    def set_epoch(self, epoch):
        """Allows updating the epoch for reproducible truncated sampling."""
        self.epoch = epoch

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_path):
        """Read image safely and apply transforms."""
        img_path = Path(img_path)
        if not img_path.is_absolute():
            img_path = Path(self.data_path) / img_path 

        with open(img_path, "rb") as f:
            img_data = f.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        if self.transform:
            img = self.transform(img)
        return img


    def _sample_label(self, idx):
        """Sample label from truncated normal for training."""
        row = self.df.iloc[idx]

        lower = row.iloc[self.target_lower_idx].values.astype(float)
        upper = row.iloc[self.target_upper_idx].values.astype(float)
        mean  = row.iloc[self.target_median_idx].values.astype(float)
        std   = row.iloc[self.target_std_idx].values.astype(float)
        
        seed = self.base_seed + self.epoch + idx
        return truncated_normal(mean, std, lower, upper, size=1, seed=seed)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {}

        # --- Image ---
        img_rel_path = row["Image"]
        img_full_path = Path(self.data_path) / img_rel_path 
        sample["image"] = self._load_image(img_full_path)

        # --- Geo features ---
        if self.modality == 'image_geo' or self.geo_location:
            geo = row.iloc[self.geo_idx].values.astype(float)
            sample["geo"] = torch.tensor(geo, dtype=torch.float)

        # --- Labels ---
        if self.phase == 'train':
            label = self._sample_label(idx)
        else:
            label = row.iloc[self.target_label_idx].values.astype(float)

        # Apply transformation to labels
        if self.phase != 'test':
            label = self.target_transform.transform(label.reshape(1, -1))

        sample["label"] = torch.tensor(label.reshape(-1), dtype=torch.float)

        return sample
