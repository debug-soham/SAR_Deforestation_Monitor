# src/data/ingest_gee.py

import os
import sys

import ee
import yaml

# Adjust path to import our custom logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.logger import get_logger

logger = get_logger("GEE_Ingestion")


def load_config(config_path="configs/pipeline_config.yaml"):
    """Loads pipeline parameters from the YAML config."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def initialize_ee():
    """Authenticates and initializes the Google Earth Engine API."""
    try:
        ee.Initialize(project="sar-deforestation-monitor")
        logger.info("Successfully connected to Google Earth Engine.")
    except Exception as e:
        logger.error(f"Earth Engine initialization failed: {e}")
        logger.error("Try running 'earthengine authenticate --force' in your terminal.")
        sys.exit(1)


def get_sar_composite(roi, start_date, end_date):
    """Fetches and processes Sentinel-1 SAR imagery for a given region and timeframe."""
    logger.info(f"Querying Sentinel-1 imagery from {start_date} to {end_date}...")

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
    )

    # Calculate median backscatter to filter out temporary atmospheric/moisture noise
    composite = collection.select(["VV", "VH"]).median().clip(roi)
    return composite


def export_to_drive(image, region_name, folder_name, scale):
    """Triggers a cloud-based export of the tensor to Google Drive."""
    filename = f"SAR_{region_name}_baseline"
    logger.info(f"Dispatching export task to GEE backend: {filename}")

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        folder=folder_name,
        fileNamePrefix=filename,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    logger.info(
        f"Task started. Track progress at: https://code.earthengine.google.com/tasks"
    )


def main():
    logger.info("Starting SAR Data Ingestion Pipeline...")

    # 1. Load config and authenticate
    config = load_config()
    initialize_ee()

    # 2. Parse geographical and temporal parameters
    bbox = config["target_region"]["bbox"]
    roi = ee.Geometry.Rectangle(bbox)
    start = config["timeframe"]["baseline_start"]
    end = config["timeframe"]["baseline_end"]

    # 3. Build the composite tensor
    sar_image = get_sar_composite(roi, start, end)

    # 4. Export the data
    export_to_drive(
        image=sar_image,
        region_name=config["target_region"]["name"],
        folder_name=config["project"]["export_folder"],
        scale=config["timeframe"]["resolution_meters"],
    )


if __name__ == "__main__":
    main()
