# ORIGEN — Automated Oil Spill Detection & Spatiotemporal Vessel Correlation

> **Smart India Hackathon 2026 (SIH 2026)**  
> **Problem Statement ID:** `SIH26143`  
> **Title:** Leveraging satellite imagery to determine oil spills at sea along with AIS data correlations to identify vessel responsible for spill  
> **Theme:** Space Technology | **Category:** Software  

---

## 🌊 Overview

**ORIGEN** is an automated maritime intelligence platform designed to detect marine oil slicks from Synthetic Aperture Radar (SAR) satellite imagery and correlate them with historical Automatic Identification System (AIS) vessel tracking data.

Unlike traditional manual methods that take **24–48 hours** to analyze static imagery, **ORIGEN** automates the entire investigation pipeline in **< 15 minutes** — reversing ocean drift to find the spill origin, identifying AIS transponder blackout gaps, and generating evidence-based vessel attribution reports.

---

## 🚨 The Problem

* **Illegal Discharges**: Ships illegally discharge toxic bilge water (viscous heavy fuel oil) via "magic pipes", bypassing international MARPOL regulations before detection occurs.
* **Moving Slicks**: Ocean currents move slicks away from the original discharge location, obscuring the source vessel.
* **Fragmented Investigations**: Authorities lack integrated tools to correlate satellite radar images with ocean current models and AIS vessel tracks.
* **High Manual Effort**: Traditional GIS analysis takes 24–48 hours, resulting in lost evidence and unpunished polluters.

---

## ✨ Why ORIGEN is Different

* 🛰️ **Goes Beyond Detection**: Segments the oil slick *and* tracks backward in time to locate the source vessel.
* ⏪ **Reverse Drift Hindcasting**: Reverses ocean drift vectors to estimate *where* and *when* the spill was released.
* 🚢 **Detects AIS Transponder Blackouts**: Flags suspicious vessels turning off AIS transponders ($\ge 2.0\text{h}$ ping gaps) during the estimated release window.
* 🎯 **Evidence-Based Vessel Ranking**: Scores candidate vessels using multi-source spatial-temporal deltas rather than assuming the nearest ship.
* 📄 **Downloadable Evidence Packages**: Generates audit-ready JSON and plain-text investigation summary packages.

---

## 🔄 Technical Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. DETECT   │ ──> │   2. TRACE   │ ──> │ 3. CORRELATE │
│ Sentinel-1   │     │ Drift Vector │     │  AIS Tracks  │
│ SAR Filtering│     │ Hindcasting  │     │ & Gap Flags  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
┌──────────────┐     ┌──────────────┐             │
│ 6. DASHBOARD │ <── │ 5. EVIDENCE  │ <───────────┘
│ Interactive  │     │ Summary &    │     ┌──────────────┐
│ Web Report   │     │ Candidate    │ <── │ 4. ATTRIBUTE │
│ Downloads    │     │ Attribution  │     │ Source Replay│
└──────────────┘     └──────────────┘     └──────────────┘
```

1. **DETECT (SAR Processing)**: Converts raw radar amplitude to decibels ($10 \cdot \log_{10}$), applies $5 \times 5$ median speckle noise filtering, and extracts slick regions using Otsu thresholding.
2. **TRACE (Drift Simulation)**: Computes forward particle trajectories and backward origin hindcasting to render a shaded **Probable Origin Region**.
3. **CORRELATE (AIS Intelligence)**: Runs a 5-stage pipeline (`All Records` $\rightarrow$ `Time Filter` $\rightarrow$ `Region Filter` $\rightarrow$ `Track-Quality Check` $\rightarrow$ `Candidates`) and flags AIS transponder gaps.
4. **ATTRIBUTE (Candidate Source Replay)**: Simulates forward drift from historical vessel coordinates to evaluate physical spatial overlap against observed slick masks.
5. **EVIDENCE (Dynamic Summary)**: Evaluates physical consistency (`High`, `Moderate`, `Low`) and generates evidence summaries based strictly on computed metrics.
6. **DASHBOARD (Web Application)**: Displays real SAR benchmark imagery, backscatter histograms, diagnostic error maps, and downloadable report packages.

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **AI & SAR Processing** | Python, PyTorch / U-Net, OpenCV, Rasterio, NumPy, SciPy |
| **Drift & Ocean Physics** | Vector displacement model (OpenDrift / Copernicus Marine framework) |
| **Vessel & Geospatial Intelligence** | GeoPandas, Shapely, Pandas, AIS Tracking Engine |
| **Web Dashboard** | Streamlit, Matplotlib, Pillow |
| **Authentication & Security** | SHA-256 password hashing, session state authorization |

---

## ⚡ Quick Start & Setup

### Prerequisites

* Python 3.8 or higher
* `pip` package manager

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Amulya

# Create & activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Launch the Web Application

```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🔑 Demo Login Credentials

You can sign in with any of the pre-configured accounts or create a new user via the **Create Account** tab:

| Role | Username | Password |
| :--- | :--- | :--- |
| **SAR Analyst** | `analyst` | `sar2026` |
| **Administrator** | `admin` | `admin123` |

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/
│   │   ├── dataset/         # Real SAR Oil Slick Benchmark Dataset (8,070 images & masks)
│   │   │   ├── images/      # SAR patch images (train / val)
│   │   │   └── masks/       # Ground-truth binary masks (train / val)
│   │   └── sample_vv.tif    # GeoTIFF raster sample
│   ├── sample_ais.csv       # Synthetic AIS vessel track records
│   └── users.json           # Secured user credentials store
├── src/
│   ├── auth.py              # User authentication & SHA-256 password hashing
│   ├── preprocess.py        # GeoTIFF loading, dB conversion, speckle filtering, normalization
│   ├── detect.py            # Otsu thresholding & morphological opening/closing
│   ├── fingerprint.py       # Geometric feature extraction (area, centroid, orientation, ratio)
│   ├── drift.py             # Vector drift simulation & candidate source replay
│   ├── ais_filter.py        # AIS track filtering & transponder blackout gap detection
│   ├── visualize.py         # Semi-transparent overlay & diagnostic error mapping
│   ├── evaluate.py          # Segmentation validation (IoU, Dice, Precision, Recall, Accuracy)
│   └── report.py            # Structured JSON & plain-text investigation report generator
├── app.py                   # Streamlit web application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📊 Benchmark Dataset & Performance

* **Validation Dataset**: 1,615 real PALSAR SAR scenes with ground-truth binary masks ($256 \times 256$ pixels).
* **Quantitative Validation Metrics**:
  * **IoU (Jaccard Index)**: `38.9%`
  * **Dice Score (F1)**: `56.0%`
  * **Precision**: `94.3%`
  * **Recall (Sensitivity)**: `39.8%`
  * **Pixel Accuracy**: `87.6%`

---

## 👥 SIH 2026 Submission

* **Team Name**: TECH6
* **Problem Statement**: SIH26143 — Space Technology
* **Core Value**: Enables rapid response to marine pollution, protects coastal fisheries, and empowers maritime authorities with traceable forensic evidence.
