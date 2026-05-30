# src/features/feature_engineering.py

import os
import sys

import numpy as np

# Import the native Rust module
try:
    import sar_core
except ImportError:
    print(
        "Error: sar_core module not found. Run 'maturin develop --release' in the sar_core directory."
    )
    sys.exit(1)


def calculate_sar_features(vv_matrix: np.ndarray, vh_matrix: np.ndarray) -> tuple:
    """
    Generates engineered spatial features for the XGBoost model by piping
    raw tensors through the Rust-accelerated Lee Filter.
    """
    print("Applying Rust-optimized Lee Filter...")

    # The Rust function natively accepts and returns NumPy arrays
    clean_vv = sar_core.fast_lee_filter(vv_matrix, window_size=5, noise_variance=0.1)
    clean_vh = sar_core.fast_lee_filter(vh_matrix, window_size=5, noise_variance=0.1)

    print("Calculating VV/VH Polarization Ratio...")

    # Calculate ratio, protecting against division by zero
    vv_vh_ratio = np.divide(
        clean_vv, clean_vh, out=np.zeros_like(clean_vv), where=clean_vh != 0
    )

    return clean_vv, clean_vh, vv_vh_ratio
