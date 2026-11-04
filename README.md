# AquaSentinel Flood Detector

> **An AI‑powered pipeline for detecting flooding from satellite imagery**

![Processing Overview](C:/Users/navde/PROJECTS/datanova/ai4g-flood/images/ASF_On_Demand_Processing_Step_1.png)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Workflow](#architecture--workflow)
3. [Installation](#installation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Data & Models](#data--models)
6. [Visualisation & Results](#visualisation--results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

AquaSentinel is a research‑grade toolkit that combines state‑of‑the‑art deep‑learning models with open‑source geospatial utilities to automatically detect flood extents from satellite imagery (e.g., Planetary Computer, Sentinel‑2).  The purpose is to provide rapid,reproducible flood maps that can be ingested by humanitarian agencies, disaster‑response teams, or downstream GIS workflows.

**Key features**:

- **End‑to‑end pipeline** – from image acquisition, preprocessing, model inference, to vectorised flood masks.
- **Modular design** – each step lives in `src/` as reusable Python modules.
- **Pre‑trained model** – the `ai4g_sar_model.ckpt` checkpoint ships with the repository.
- **Extensible** – swap in custom models or data sources with minimal code changes.

The project is built with Python 3.10+, relies on `torch`, `rasterio`, `geopandas` and a few utilities from the `src/utils` package.

---

## Architecture & Workflow

```mermaid
flowchart LR
    A[Download Images] --> B[Pre‑process (utils/image_processing.py)]
    B --> C[Run Model (utils/model.py)]
    C --> D[Post‑process & Vectorise (utils/flood_dataset.py)]
    D --> E[Export GeoJSON / Raster]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
```

The pipeline is orchestrated by two entry‑point scripts in `src/`:

- `run_flood_detection_downloaded_images.py` – works on a local folder of already‑downloaded images.
- `run_flood_detection_planetary_computer.py` – pulls imagery directly from Microsoft Planetary Computer.

Both scripts share the same core utilities located under `src/utils/`:

- **`image_processing.py`** – handles tiling, resampling, and band selection.
- **`model.py`** – loads the pre‑trained SAR model and runs inference.
- **`flood_dataset.py` & `flood_data_module.py`** – convert model logits into binary flood masks, apply morphological cleanup, and export results.

![Processing Steps Detail](C:/Users/navde/PROJECTS/datanova/ai4g-flood/images/ASF_On_Demand_Processing_Step_2.png)

---

## Installation

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/navdeep-r/AquaSentinel.git
cd AquaSentinel

# Create a conda environment (recommended)
conda env create -f environment.yml
conda activate AquaSentinel

# Or use pip with the pinned requirements
pip install -r requirements.txt
```

> **Note**: The repository ships a `pyproject.toml` for optional build‑system integration.

### Dependencies Highlights

- `torch>=2.0` – deep‑learning framework (CPU/GPU auto‑detect).
- `rasterio`, `geopandas` – geospatial raster/vector IO.
- `planetary-computer` – for on‑the‑fly image retrieval (used by the planetary‑computer script).

---

## Running the Pipeline

### 1️⃣ Process a directory of local images

```bash
python src/run_flood_detection_downloaded_images.py \
    --input-dir ./data/sample_images \
    --output-dir ./outputs \
    --model-path ./models/ai4g_sar_model.ckpt
```

### 2️⃣ Pull images from Planetary Computer

```bash
python src/run_flood_detection_planetary_computer.py \
    --bbox "-120.0,35.0,-118.5,36.5" \
    --date "2023-08-01/2023-08-07" \
    --output-dir ./outputs \
    --model-path ./models/ai4g_sar_model.ckpt
```

Both scripts emit:
- A raster (`.tif`) of the flood probability.
- A binary flood mask (`.tif`).
- A vectorised GeoJSON of flood polygons.

---

## Data & Models

- **Satellite imagery** – the pipeline expects multi‑band GeoTIFFs (e.g., Sentinel‑2, PlanetScope).  The `src/data` folder contains a small sample of country boundaries used for masking.
- **Pre‑trained model** – `models/ai4g_sar_model.ckpt` is a SAR‑based flood classifier trained on the *FloodNet* dataset.  It can be swapped for any `torch.nn.Module` that outputs a single‑channel probability mask.

If you wish to train your own model, see the `merge.py` script for a quick example of how to combine multiple checkpoints.

---

## Visualisation & Results

The exported GeoJSON can be opened in QGIS, ArcGIS, or any web‑map library (e.g., Leaflet, kepler.gl).  For quick inspection, run:

```bash
python -m src.utils.flood_dataset \
    --raster ./outputs/flood_prob.tif \
    --threshold 0.5 \
    --visualise
```

This will pop up a Matplotlib window overlaying the flood mask on the original imagery.

---

## Contributing

We welcome contributions!  Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b my‑feature`).
3. Ensure the test suite passes (`pytest`).
4. Open a Pull Request with a clear description of the change.

Please adhere to the existing code style (PEP‑8) and update the documentation when adding new functionality.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

*Happy flooding detection!*