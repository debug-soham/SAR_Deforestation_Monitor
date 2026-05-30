# src/models/train_xgboost.py

import os
import sys

import numpy as np
import rasterio
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Import our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.feature_engineering import calculate_sar_features
from utils.logger import get_logger

logger = get_logger("XGBoost_Engine")


def load_geotiff(filepath):
    """Reads a Sentinel-1 GeoTIFF and extracts the VV and VH bands."""
    logger.info(f"Loading SAR data from {filepath}...")
    with rasterio.open(filepath) as src:
        # Assuming Band 1 is VV and Band 2 is VH based on our GEE export
        vv_matrix = src.read(1)
        vh_matrix = src.read(2)
    return vv_matrix, vh_matrix


def prepare_tabular_data(vv, vh, labels=None):
    """
    Passes spatial matrices through the Rust kernel, then flattens
    them into a 1D tabular dataset for XGBoost.
    """
    # 1. High-Performance Rust Preprocessing
    clean_vv, clean_vh, vv_vh_ratio = calculate_sar_features(vv, vh)

    # 2. Flatten 2D matrices into 1D arrays (Pixels -> Rows)
    X = np.column_stack((clean_vv.flatten(), clean_vh.flatten(), vv_vh_ratio.flatten()))

    # 3. Handle Labels (If training)
    y = labels.flatten() if labels is not None else None

    return X, y


def train_model(X, y):
    """Trains the XGBoost Classifier on the engineered SAR features."""
    logger.info("Splitting data into training and validation sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Initializing XGBoost Engine...")
    # These hyperparameters are tuned for highly imbalanced geospatial data
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=10,  # Assuming deforestation is rare compared to healthy forest
        n_jobs=-1,  # Use all available CPU cores
        random_state=42,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    logger.info("Evaluating model...")
    preds = model.predict(X_test)
    logger.info(f"\n{classification_report(y_test, preds)}")

    return model


if __name__ == "__main__":
    # --- PROTOTYPE PIPELINE EXECUTION ---

    # 1. Define paths to your downloaded GEE data
    # (Replace with your actual downloaded file path)
    sar_filepath = "data/raw/SAR_Rondonia_Brazil_baseline.tif"

    if not os.path.exists(sar_filepath):
        logger.error(f"File not found: {sar_filepath}. Run ingest_gee.py first.")
        sys.exit(1)

    # 2. Load the raw radar matrices
    vv_raw, vh_raw = load_geotiff(sar_filepath)

    # 3. Generate a mock ground-truth mask for testing the pipeline
    # In production, this would be a loaded GeoTIFF containing actual deforestation labels (1=Deforested, 0=Forest)
    logger.info("Generating synthetic labels for pipeline validation...")
    mock_labels = np.random.choice([0, 1], size=vv_raw.shape, p=[0.95, 0.05])

    # 4. Process through Rust and format for ML
    X, y = prepare_tabular_data(vv_raw, vh_raw, mock_labels)

    # 5. Train the System
    trained_model = train_model(X, y)
    logger.info("Pipeline executed successfully. Model is ready for deployment.")
