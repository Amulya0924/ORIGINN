# ORIGIN — Intelligent Marine Oil Spill Detection & Vessel Attribution

<div align="center">

[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-blue.svg)](https://sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/Problem%20Statement-SIH26143-orange.svg)](https://sih.gov.in/)
[![Theme](https://img.shields.io/badge/Theme-Disaster%20Management-purple.svg)]()
[![Category](https://img.shields.io/badge/Category-Software-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Prototype%20%2F%20PoC-yellow.svg)]()

**“Detect the spill. Reconstruct the event. Test the source.”**

*Automated SAR Satellite Slick Detection, Vector Drift Origin Hindcasting, AIS Transponder Blackout Analysis, and Counterfactual Source Replay Simulation.*

</div>

---

## 📋 Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Problem Statement & System Innovation](#2-problem-statement--system-innovation)
- [3. End-to-End System Architecture](#3-end-to-end-system-architecture)
- [4. Core Technical Modules](#4-core-technical-modules)
- [5. Mathematical & Algorithmic Formulations](#5-mathematical--algorithmic-formulations)
- [6. Interactive Web Workspace Guide](#6-interactive-web-workspace-guide)
- [7. Quantitative Evaluation & Benchmark Results](#7-quantitative-evaluation--benchmark-results)
- [8. Honest Prototype Limitations & Technical Transparency](#8-honest-prototype-limitations--technical-transparency)
- [9. System Roadmap & Future Enhancements](#9-system-roadmap--future-enhancements)
- [10. Quick Start & Installation Guide](#10-quick-start--installation-guide)

---

## 1. Executive Summary

Marine oil spills from illegal vessel discharges—such as toxic bilge water dumping via unauthorized "magic pipes"—cause severe ecological harm to coastal ecosystems and marine life. Traditional manual surveillance relies on fragmented tools: analysts inspect satellite images, separately query AIS tracking data, and manually calculate ocean drift. This manual GIS workflow typically requires **24 to 48 hours**, during which ocean currents disperse the slick and evidence dissipates.

**ORIGEN** (*Intelligent Marine Oil Spill Detection & Vessel Attribution*) unifies satellite radar analysis, trajectory reconstruction, and vessel intelligence into a streamlined **< 15 minute workflow**:
1. **Detects** dark capillary wave-damped oil slicks in Sentinel-1/PALSAR SAR imagery.
2. **Reconstructs** historical slick movement backward in time to estimate a **Probable Origin Region**.
3. **Correlates** historical AIS vessel tracks to flag suspicious vessels and transponder blackout gaps ($\ge 2.0\text{h}$).
4. **Tests** candidate vessels via counterfactual forward drift replay simulation to rank attribution consistency.
5. **Generates** audit-ready JSON and plain-text investigation report packages.

---

## 2. Problem Statement & System Innovation

### SIH 2026 Context
* **Problem Statement ID:** `SIH26143`
* **Title:** *Leveraging satellite imagery to determine oil spills at sea along with AIS data correlations to identify vessel responsible for spill*
* **Domain:** Space Technology / Maritime Intelligence

### Key System Innovations

```
                       TRADITIONAL MANUAL WORKFLOW
 ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
 │ Satellite      │ ──> │ Manual GIS     │ ──> │ Nearest-Vessel │  ⏱️ 24 - 48 Hours
 │ Image Lookup   │     │ Drift Query    │     │ Assumption     │  ❌ Low Attribution Confidence
 └────────────────┘     └────────────────┘     └────────────────┘

                         ORIGEN INTEGRATED PIPELINE
 ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
 │ Automated SAR  │ ──> │ Origin Vector  │ ──> │ Counterfactual │  ⏱️ < 15 Minutes
 │ Slick Segment  │     │ Hindcasting    │     │ Replay Score   │  ✅ Physical Consistency Ranking
 └────────────────┘     └────────────────┘     └────────────────┘
```

| Challenge | Traditional Approach | ORIGEN Solution |
| :--- | :--- | :--- |
| **Slick Movement** | Static imagery analysis assumes slick stayed where detected | Reverses ocean drift vectors to estimate release origin |
| **AIS Gaps** | Transponder blackouts hide suspicious vessel activity | Automatically detects AIS gaps ($\ge 2.0\text{h}$) in estimated release window |
| **Vessel Attribution** | Assumes nearest ship at observation time is guilty | Runs counterfactual source replay simulation from historical vessel positions |
| **Audit & Legal Trail** | Fragmented manual notes | Generates downloadable JSON & plain-text investigation packages |

---

## 3. End-to-End System Architecture

The pipeline processes raw SAR satellite rasters through six sequential stages:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               ORIGEN PIPELINE WORKFLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      │                                                                             │
      ▼                                                                             ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐  ┌──────────────┐
│  1. DETECT   │ ──> │   2. TRACE   │ ──> │ 3. CORRELATE │ ──> │ 4. ATTRIBUTE │─>│ 5. EVIDENCE  │
│  (SAR Radar  │     │ (Vector Drift│     │  (AIS Track  │     │(Source Replay│  │ (Summary &   │
│ Segmentation)│     │ Hindcasting) │     │ & Gap Flags) │     │ Simulation)  │  │ Export PKG)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘  └──────────────┘
                                                                                        │
                                                                                        ▼
                                                                                 ┌──────────────┐
                                                                                 │ 6. DASHBOARD │
                                                                                 │ (Streamlit   │
                                                                                 │ Workspace)   │
                                                                                 └──────────────┘
```

1. **`DETECT` (`src/preprocess.py` + `src/detect.py`)**: Converts linear amplitude/intensity to decibels ($10\log_{10}$), filters speckle noise with a $5\times5$ median kernel, and extracts dark candidate slick masks via Otsu thresholding.
2. **`TRACE` (`src/drift.py`)**: Computes forward particle trajectories and backward origin hindcasting to estimate a shaded **Probable Origin Region**.
3. **`CORRELATE` (`src/ais_filter.py`)**: Filters AIS tracks through a 5-stage pipeline (`All Records` $\rightarrow$ `Time Filter` $\rightarrow$ `Region Filter` $\rightarrow$ `Track Check` $\rightarrow$ `Candidates`) and flags transponder blackouts ($\ge 2.0\text{h}$).
4. **`ATTRIBUTE` (`src/drift.py` + `app.py`)**: Simulates forward drift from historical candidate coordinates and overlays particles onto the observed slick mask.
5. **`EVIDENCE` (`src/report.py`)**: Formulates dynamic evidence summaries and qualitative physical consistency bands (`High`, `Moderate`, `Low`).
6. **`DASHBOARD` (`app.py`)**: Presents interactive UI controls, side-by-side visual comparisons, confusion error maps, and downloadable report files.

---

## 4. Core Technical Modules

The backend architecture is structured into decoupled Python modules in `src/`:

```
ORIGIN/
├── app.py                   # Streamlit interactive web application & UI state controller
├── requirements.txt         # Project dependency specification
├── README.md                # System documentation
├── data/
│   └── sample_ais.csv       # Demonstrational synthetic AIS tracking dataset (18 records)
└── src/
    ├── __init__.py          # Package initialization
    ├── auth.py              # User authentication, SHA-256 password hashing, session state
    ├── preprocess.py        # Rasterio GeoTIFF loading, dB conversion, 5x5 median filtering
    ├── detect.py            # Otsu automatic thresholding & 5x5 morphological opening/closing
    ├── fingerprint.py       # OpenCV geometry extraction (area, centroid, orientation angle, elongation)
    ├── drift.py             # Vector particle displacement drift simulation & counterfactual replay
    ├── ais_filter.py        # Spatial/temporal candidate filtering, Haversine distance, gap check
    ├── visualize.py         # Semi-transparent overlay blending & confusion matrix error mapping
    ├── evaluate.py          # Segmentation validation metrics (IoU, Dice, Precision, Recall, Accuracy)
    └── report.py            # Structured JSON & formatted plain-text investigation package generator
```

### Module Specifications

#### `src/preprocess.py` — SAR Image Ingestion & Denoising
* Ingests single-band VV polarimetric SAR rasters (`.tif` / `.tiff` via `rasterio`) or standard display formats (`.png` / `.jpg` via `PIL`).
* Converts linear pixel intensity values to decibels: $\text{dB} = 10 \cdot \log_{10}(\max(x, 10^{-10}))$.
* Reduces multiplicative radar speckle noise using a $5 \times 5$ median spatial filter (`scipy.ndimage.median_filter`).
* Normalizes backscatter values to a 0–255 `uint8` range for matrix operations and display rendering.

#### `src/detect.py` — Automatic Thresholding & Morphological Cleanup
* Implements Otsu's thresholding algorithm (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) to segment low-backscatter (dark) oil slick regions from brighter ocean background.
* Supports interactive sensitivity adjustments via an analyst threshold offset slider (`effective_thresh = np.clip(otsu_thresh + offset, 0, 255)`).
* Applies morphological filtering using a $5 \times 5$ rectangular kernel:
  * **Morphological Opening**: Eliminates small isolated false-positive noise pixels.
  * **Morphological Closing**: Fills interior voids within detected slick masks.

#### `src/fingerprint.py` — Geometric Slick Feature Extraction
* Computes real spatial properties from binary predicted slick masks:
  * **Area**: Total detected slick pixels and percentage of scene coverage.
  * **Centroid**: Spatial center $(C_x, C_y)$ computed via spatial moments (`cv2.moments`).
  * **Bounding Box**: Minimum-area bounding box length, width, and orientation angle ($\theta$) via `cv2.minAreaRect`.
  * **Shape Classification**: Calculates elongation ratio ($\text{Length} / \text{Width}$) to categorize slicks as `"Elongated"` ($\ge 1.8$) or `"Compact"`.

#### `src/drift.py` — Vector Drift Modeling & Counterfactual Replay
* Executes particle displacement drift simulations:
  * **Forward Mode**: Predicts particle trajectory over 1–48 hours given wind/current heading and speed.
  * **Backward Mode**: Hindcasts movement backward in time to project a shaded **Probable Origin Region** circle around the origin centroid.
  * **Counterfactual Replay**: Simulates forward drift starting from a candidate vessel's historical AIS position and overlays particles onto the observed slick mask.

#### `src/ais_filter.py` — AIS Track Correlation & Transponder Gap Detection
* Filters candidate vessel tracks by time proximity ($\le 4.0\text{h}$) and spatial proximity ($\le 15.0\text{km}$) using Haversine distance calculations.
* Detects transponder blackout gaps by computing consecutive ping time deltas per vessel ($\Delta t \ge 2.0\text{h}$).
* Categorizes threat status dynamically (`High`, `Medium`, `Low`) based on actual spatial and temporal deltas.

#### `src/report.py` — Evidence Package Generation
* Compiles session investigation findings into a structured JSON dictionary and plain-text (`.txt`) report string.
* Includes analyst metadata, geometric slick fingerprinting, environmental vector parameters, top vessel candidate metrics, and dynamic evidence summary statements.

---

## 5. Mathematical & Algorithmic Formulations

### 1. Decibel (dB) Backscatter Conversion
$$\text{dB} = 10 \cdot \log_{10}(\max(I, 10^{-10}))$$

### 2. Spatial Centroid via Image Moments
$$C_x = \frac{M_{10}}{M_{00}}, \quad C_y = \frac{M_{01}}{M_{00}}$$
where $M_{pq} = \sum_{x} \sum_{y} x^p y^q I(x, y)$.

### 3. Great-Circle Distance (Haversine Formula)
$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
$$d = 2 \cdot R \cdot \operatorname{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$
where $\phi$ is latitude, $\lambda$ is longitude, and $R = 6371.0\text{ km}$.

### 4. Segmentation Validation Metrics
$$\text{IoU (Jaccard Index)} = \frac{|P \cap G|}{|P \cup G|}$$
$$\text{Dice (F1 Score)} = \frac{2 \cdot |P \cap G|}{|P| + |G|}$$
$$\text{Precision} = \frac{|P \cap G|}{|P|}, \quad \text{Recall} = \frac{|P \cap G|}{|G|}$$

---

## 6. Interactive Web Workspace Guide

The Streamlit web application (`app.py`) provides an interactive interface for maritime analysts:

```
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │ 🛰️ ORIGIN Investigation Workspace                                                     │
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ Data Mode: [● Real Benchmark Dataset Explorer]  [○ Upload Custom SAR File]            │
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ 🔍 Slick Fingerprint: Area: 3,888 px (5.93%) | Centroid: (128.0, 128.0) | Angle: -59°   │
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ [🖼️ Visual Comparison] [🧩 Error Diagnostic] [📈 dB Analytics] [🌊 Drift Reconstruction]│
 │ [🚢 AIS Investigation] [📥 Download Report Package]                                   │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

### Pre-Configured Demo Credentials

| Role | Username | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **SAR Marine Analyst** | `analyst` | `sar2026` | Full investigation workspace & report export |
| **System Administrator** | `admin` | `admin123` | Full administrative & workspace access |

---

## 7. Quantitative Evaluation & Benchmark Results

ORIGIN was evaluated against **1,615 real validation SAR scenes** from the Synthetic Aperture Radar Oil Slick Benchmark Dataset ($256 \times 256$ pixel patches with paired ground-truth binary masks):

| Metric | Benchmark Score | Description |
| :--- | :--- | :--- |
| **IoU (Jaccard Index)** | **38.9%** | Spatial overlap between predicted mask and ground truth |
| **Dice Coefficient (F1)** | **56.0%** | Harmonic mean of precision and recall |
| **Precision** | **94.3%** | Proportion of detected slick pixels that are true oil slicks |
| **Recall (Sensitivity)** | **39.8%** | Proportion of ground-truth slick pixels correctly detected |
| **Pixel Accuracy** | **87.6%** | Overall pixel classification accuracy (slick vs background ocean) |

---

## 8. Honest Prototype Limitations & Technical Transparency

To maintain technical credibility during evaluation, the current implementation status and limitations are outlined below:

> [!CAUTION]
> ### Prototype vs Production Technical Audit
> 
> 1. **Simplified Vector Drift Model**:
>    - *Current Implementation*: Constant-vector particle displacement based on manual wind/current heading and speed sliders.
>    - *Limitation*: Does not model dynamic hydrodynamic ocean currents, Stokes drift, or oil weathering (evaporation, emulsification).
> 
> 2. **Demonstrational AIS Dataset**:
>    - *Current Implementation*: Ingests local synthetic AIS records (`data/sample_ais.csv`, 18 vessel tracks).
>    - *Limitation*: Does not connect to live commercial AIS APIs (Spire, MarineTraffic) or live satellite AIS streams.
> 
> 3. **Rule-Based Segmentation Baseline**:
>    - *Current Implementation*: Automated Otsu thresholding + morphological filtering with manual sensitivity offset controls.
>    - *Limitation*: High-contrast radar dark formation segmentation; deep learning U-Net model inference integration is in progress.
> 
> 4. **User Storage**:
>    - *Current Implementation*: Local encrypted JSON file store (`data/users.json`) with SHA-256 password hashing.
>    - *Limitation*: Uses local file storage rather than an enterprise PostgreSQL/PostGIS database instance.

---

## 9. System Roadmap & Future Enhancements

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                   ORIGEN ROADMAP                                      │
├───────────────────────────┬───────────────────────────┬───────────────────────────────┤
│ PHASE 1: PROTOTYPE        │ PHASE 2: PHYSICS & AI     │ PHASE 3: ENTERPRISE           │
│ (Current Implementation)  │ (In Progress)             │ (Planned)                     │
├───────────────────────────┼───────────────────────────┼───────────────────────────────┤
│ • Otsu thresholding       │ • PyTorch U-Net inference │ • Live Spire AIS API stream   │
│ • Vector drift model      │ • OpenDrift Lagrangian    │ • PostGIS spatial database    │
│ • Synthetic AIS sample    │   ocean physics           │ • Automated agency email      │
│ • Download JSON/TXT       │ • Copernicus Marine currents │ alerts & legal audit logs   │
└───────────────────────────┴───────────────────────────┴───────────────────────────────┘
```

---

## 10. Quick Start & Installation Guide

### Prerequisites
* Python 3.8 or higher
* `pip` package manager

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/prash7117/ORIGIN.git
   cd ORIGIN
   ```

2. **Set Up Virtual Environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Web Application**:
   ```bash
   streamlit run app.py
   ```

5. **Access Dashboard**:
   Open **`http://localhost:8501`** in your browser and log in with `analyst` / `sar2026`.

---

<div align="center">

**Smart India Hackathon 2026 — Problem Statement SIH26143**  
*Team TECH6 | Disaster Management Theme*

</div>
