# SAR Deforestation Monitor

A high-performance geospatial machine learning pipeline designed to detect canopy loss and illegal logging using Synthetic Aperture Radar (SAR). 

This project bypasses the limitations of optical satellite imagery (which is easily blocked by cloud cover) by utilizing Sentinel-1 C-band radar backscatter. It combines cloud-based data ingestion, bare-metal Rust optimizations for spatial filtering, and classical machine learning (XGBoost) for structural classification.

## Architecture Overview

The pipeline is split into three primary modules, strictly separating data ingestion, high-performance mathematical preprocessing, and machine learning logic.

1. **Google Earth Engine (GEE) Ingestion:** A Python pipeline that queries the GEE API to pull radiometrically calibrated `VV` and `VH` polarization tensors over targeted deforestation hotspots (e.g., Rondônia, Brazil), using median compositing to reduce transient atmospheric noise.
2. **Rust-Accelerated Spatial Filtering (`sar_core`):** A custom native Python module written in **Rust** (compiled via PyO3/Maturin). It utilizes the `Rayon` crate to automatically multi-thread a localized Lee Speckle Filter across massive geospatial matrices, bypassing the severe latency bottlenecks of Python-based spatial loops.
3. **XGBoost Classification Engine:** A flattened tabular data architecture that processes the cleaned radar features and engineered polarization ratios to learn non-linear thresholds associated with clear-cut terrain.

## Installation & Setup

**Prerequisites:** * Python 3.12 (Strict requirement for pre-compiled `rasterio` GDAL binaries on Windows)
* Rust & Cargo (`rustup`)
* Google Earth Engine Account (Registered for non-commercial/academic use)

```bash
# 1. Clone the repository
git clone https://github.com/debug-soham/SAR_Deforestation_Monitor.git
cd SAR_Deforestation_Monitor

# 2. Create and activate a Python 3.12 virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
pip install maturin

# 4. Compile the Rust native module
cd sar_core
maturin develop --release
cd ..
```

## Execution

1. Authenticate your Earth Engine credentials:
    ```bash
    earthengine authenticate --force
    ```

2. Update your specific Google Cloud Project ID in `src/data/ingest_gee.py`, then run the ingestion pipeline to generate the baseline GeoTIFF:
    ```bash
    python src/data/ingest_gee.py
    ```

3. Once the `.tif` file is downloaded to `data/raw/`, execute the end-to-end training pipeline:
    ```bash
    python src/models/train_xgboost.py
    ```

## Roadmap & Ongoing Work

This repository currently serves as a fully functional **architectural prototype**. The data pipeline, Rust extensions, and ML framework are heavily optimized and structurally complete.

**Next Steps (Data Integration):**
Currently, the XGBoost engine is validated using synthetic probability masks to ensure the flattening and tensor-processing pipelines execute without memory leaks or architectural faults.

The immediate next phase of development involves replacing the synthetic validation masks with official **PRODES Deforestation Masks** (the Brazilian government's official ground-truth dataset) or the **Hansen Global Forest Change** dataset to begin active model tuning on physical canopy loss signatures.
