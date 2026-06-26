<div align="center">

# FloodSense-PK

### National Flood Intelligence & Early Warning System for Pakistan

<a href="https://floodsense-pk.streamlit.app/">
  <img src="https://img.shields.io/badge/Live_Demo-floodsense--pk.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo on Streamlit Cloud"/>
</a>

<br/><br/>

<img src="https://img.shields.io/badge/IoU-0.5503-brightgreen?style=for-the-badge&logo=target" alt="IoU badge"/>
&nbsp;
<img src="https://img.shields.io/badge/PyTorch-UNet_ResNet34-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch badge"/>
&nbsp;
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit badge"/>

<br/>

<img src="https://img.shields.io/badge/Google_Earth_Engine-4285F4?style=for-the-badge&logo=google-earth-engine&logoColor=white" alt="GEE badge"/>
&nbsp;
<img src="https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=google-gemini&logoColor=white" alt="Gemini badge"/>
&nbsp;
<img src="https://img.shields.io/badge/Sentinel--1-SAR-0078D4?style=for-the-badge&logo=satellite" alt="Sentinel-1 badge"/>
&nbsp;
<img src="https://img.shields.io/badge/Landsat--5-2010_Baseline-2E7D32?style=for-the-badge&logo=nasa" alt="Landsat-5 badge"/>

</div>

<p align="center">
  <strong>High-fidelity flood monitoring combining Satellite Radar AI, Historical Benchmarks, and Real-time Hydraulic Data.</strong><br/>
  Built for disaster management authorities with data-driven response when monsoon clouds block optical satellites.
</p>

```mermaid
flowchart TB
    A[Sentinel-1 SAR Image] --> B[U-Net ResNet34]
    B --> C[Flood Probability Map]
    C --> D[Weighted Risk Score 1 to 10]
    E[Landsat-5 MNDWI 2010] --> F[Delta vs Benchmark]
    D --> G[Gemini or Groq Tactical Report]
    F --> G
```

### Web App (Streamlit)

<p align="center">
  <img src="public/dashboard/screencapture-localhost-8501-2026-05-29-23_47_45.png" alt="FloodSense-PK Streamlit web dashboard overview" width="920" />
</p>

<p align="center"><em>Web dashboard: District, Province, and National analysis. <a href="https://floodsense-pk.streamlit.app/">Try the live demo</a></em></p>

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [System Architecture](#system-architecture)
- [Executive Dashboard](#executive-dashboard)
- [Mobile Alert App (Flet)](#mobile-alert-app-flet)
- [Satellite Intelligence](#satellite-intelligence)
- [Deep Learning Engine](#deep-learning-engine)
- [Results and Visual Outputs](#results-and-visual-outputs)
- [Historical Benchmarking (2010)](#historical-benchmarking-2010)
- [Hydraulic Command Center](#hydraulic-command-center)
- [Strategic AI Insights](#strategic-ai-insights)
- [Risk Scoring](#risk-scoring)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Project Tools & Dependencies](#project-tools--dependencies)
- [References](#references)

---

## Overview

**FloodSense-PK** is an end-to-end flood intelligence platform for Pakistan. It fuses three independent evidence streams:

| Layer    | Source                          | What it answers                                               |
| -------- | ------------------------------- | ------------------------------------------------------------- |
| **Now**  | Sentinel-1 SAR + U-Net ResNet34 | Where is water _right now_, through clouds and at night?      |
| **Then** | Landsat-5 MNDWI (2010 baseline) | How does today compare to the **2010 Great Pakistan Floods**? |
| **Live** | FFD river discharge scraper     | Are upstream barrages in NORMAL, HIGH, or EXTREME status?     |

FloodSense-PK ships as **two frontends** that share the same flood intelligence engine:

| App            | Entry point                                                            | Best for                                           |
| -------------- | ---------------------------------------------------------------------- | -------------------------------------------------- |
| **Web App**    | `streamlit_app.py` ([live demo](https://floodsense-pk.streamlit.app/)) | Full executive dashboard: maps, rivers, AI reports |
| **Mobile App** | `mobile_app/main.py`                                                   | Personal district alerts on the go (Flet)          |
| **CLI**        | `main.py`                                                              | Headless batch district analysis                   |

### Key capabilities (shared engine)

- **Multi-scale analysis**: District, Province, or National scope with dynamic GEE resolution (~80 m district / ~1000 m overview).
- **Tiled U-Net inference**: Large SAR tiles (up to 1024x1024) sliced into 256x256 patches with stitched flood masks.
- **2010 vs Current comparison**: Side-by-side historical and live flood % with **Delta Severity**.
- **Native river map** (web): Pydeck topology linking Tarbela to Sukkur to Kotri and tributary networks.
- **AI tactical reports** (web): Gemini (primary) or Groq (fallback) generate structured operational briefings.

### Model benchmark (Sen1Floods11)

| Metric              | Value                                 |
| ------------------- | ------------------------------------- |
| Best Validation IoU | **0.5503**                            |
| Best Epoch          | 18 / 60                               |
| Architecture        | U-Net + ResNet34 (ImageNet encoder)   |
| Input channels      | 3 (SAR VV, SAR VH, VH/VV ratio)       |
| Parameters          | ~24 million                           |
| Training dataset    | Sen1Floods11 (11 global flood events) |

---

## Why This Matters

```
┌─────────────────────────────────────────────────────────┐
│  5–8 million people affected by floods annually (PK)    │
│  $3–4 billion annual economic losses                    │
│  33 million displaced in the 2022 Pakistan floods       │
│  < 6 hours warning for many rural communities           │
└─────────────────────────────────────────────────────────┘
```

Traditional flood mapping fails during active monsoon events: field surveys are slow and dangerous, and **optical satellites cannot see through cloud cover**. FloodSense-PK uses **C-band SAR** that penetrates clouds and works 24/7, then grounds predictions in **2010 historical severity** and **live barrage discharge** from Pakistan's Flood Forecasting Division (FFD).

---

## System Architecture

### Platform overview (Web + Mobile)

```mermaid
flowchart TB
    Browser[Web Browser]
    Phone[Flet Mobile App]

    Web[streamlit_app.py Web App]
    Mobile[mobile_app/main.py Mobile App]
    CLI[main.py CLI]

    INF[model_inference.py]
    NDWI[ndwi.py]
    FFC[ffc_scraper.py]
    AI[ai_alerts.py]

    GEE[Google Earth Engine]
    FFD[FFD River Portal]
    GEM[Gemini API]
    GRQ[Groq API]

    Browser --> Web
    Phone --> Mobile

    Web --> INF
    Web --> NDWI
    Web --> FFC
    Web --> AI
    Mobile --> INF
    Mobile --> NDWI
    Mobile --> FFC
    CLI --> INF
    CLI --> NDWI

    INF --> GEE
    NDWI --> GEE
    FFC --> FFD
    AI --> GEM
    AI --> GRQ
```

### Web App analysis flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant GEE as Earth Engine
    participant ML as UNet Inference
    participant FFD as FFD Scraper
    participant Risk as Risk Engine
    participant AI as Gemini Groq

    User->>Streamlit: Select district and date range
    Streamlit->>GEE: Fetch 2010 MNDWI baseline
    GEE-->>Streamlit: Historical flood percent
    Streamlit->>GEE: Fetch Sentinel-1 SAR
    GEE-->>Streamlit: SAR GeoTIFF
    Streamlit->>ML: Run U-Net tiled inference
    ML-->>Streamlit: Flood mask and metrics
    Streamlit->>FFD: Scrape barrage discharge
    FFD-->>Streamlit: River status
    Streamlit->>Risk: Combine flood and river data
    Risk-->>Streamlit: Risk score 1 to 10
    Streamlit->>AI: Send metrics context
    AI-->>Streamlit: Tactical report
    Streamlit-->>User: Overview Detection Rivers AI tabs
```

<p align="center">
  <img src="public/architecture/sequence-diagram.png" alt="FloodSense-PK web app sequence diagram" width="920" />
</p>

<p align="center"><em>Static PNG fallback if Mermaid preview is unavailable</em></p>

### Mobile App flow

```mermaid
flowchart LR
    Login[Login Screen] --> Home[Home Tab]
    Register[Register Screen] --> Login
    Home --> Map[Analysis Tab]
    Home --> Profile[Profile Tab]
    Map --> Home
    Profile --> Login
```

<p align="center"><em>Mobile app uses the same GEE + U-Net + FFD engine via <code>alert_engine.py</code>. No AI report tab on mobile.</em></p>

---

## Executive Dashboard (Web App)

The **Streamlit web dashboard** runs locally at `http://localhost:8501` or on **[Streamlit Cloud](https://floodsense-pk.streamlit.app/)** with four analytical tabs after **Run analysis**.

| Tab                 | Purpose                                                  |
| ------------------- | -------------------------------------------------------- |
| **Overview**        | 2010 vs Current side-by-side, Delta Severity, risk score |
| **Detection**       | SAR + probability heatmap + unified mask, km² affected   |
| **River Flows**     | FFD status map, bar charts, inflow/outflow scatter       |
| **AI Intelligence** | Structured Gemini tactical report (web only)             |

> **Web vs Mobile:** The Streamlit web app has 4 tabs including River Flows and AI Intelligence. The Flet mobile app has 3 tabs (Home, Analysis, Profile) with login and district-based alerts.

### Overview: 2010 vs Current comparison

<p align="center">
  <img src="public/dashboard/screencapture-localhost-8501-2026-05-29-23_48_08.png" alt="Dashboard overview tab" width="920" />
</p>

### Detection: U-Net outputs and confidence

<p align="center">
  <img src="public/dashboard/screencapture-localhost-8501-2026-05-29-23_48_20.png" alt="Dashboard detection tab" width="920" />
</p>

### River Flows: FFD hydraulic network

<p align="center">
  <img src="public/dashboard/screencapture-localhost-8501-2026-05-29-23_48_33.png" alt="Dashboard river flows tab" width="920" />
</p>

---

## Mobile Alert App (Flet)

The **Flet mobile app** (`mobile_app/`) is a separate frontend from the Streamlit web dashboard. It delivers personalized district alerts, live SAR analysis, and risk scores using the same GEE + U-Net + FFD engine (via `alert_engine.py`).

| Screen              | Route       | What it shows                                              |
| ------------------- | ----------- | ---------------------------------------------------------- |
| **Login**           | `/login`    | Secure sign-in to the National Intelligence Portal         |
| **Register**        | `/register` | Create account + pick your monitoring district             |
| **District Picker** | `/register` | Scrollable list of all 148 Pakistan districts              |
| **Home**            | `/home`     | Risk score, current vs 2010 flood %, river status, refresh |
| **Analysis**        | `/map`      | Satellite flood map overlay with legend                    |
| **Profile**         | `/profile`  | User info, target district, logout                         |

### Authentication: Login and Register

<p align="center">

|                                               Login                                                |                                               Register                                                |
| :------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| <img src="public/flat%20mobile%20app/signin.png" width="300" alt="Flet mobile app login screen" /> | <img src="public/flat%20mobile%20app/signup.png" width="300" alt="Flet mobile app register screen" /> |

</p>

<p align="center"><em>Login screen and Register with username, password, and district selection</em></p>

### District selection (148 districts)

<p align="center">
  <img src="public/flat%20mobile%20app/singup2.png" width="300" alt="Flet mobile app district picker dropdown" />
</p>

<p align="center"><em>Register: scrollable district dropdown for personalized flood monitoring</em></p>

### Main app: Home, Analysis, Profile

<p align="center">

|                                               Home                                                |                                                  Analysis                                                  |                                             Profile                                             |
| :-----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
| <img src="public/flat%20mobile%20app/dashbaord.png" width="280" alt="Flet mobile app Home tab" /> | <img src="public/flat%20mobile%20app/dashboard_anly.png" width="280" alt="Flet mobile app Analysis tab" /> | <img src="public/flat%20mobile%20app/user.png" width="280" alt="Flet mobile app Profile tab" /> |

</p>

<p align="center"><em>Home: risk level and 2010 comparison. Analysis: SAR flood map. Profile: account and logout.</em></p>

### Mobile app features (not on web)

- **User authentication**: SQLite-backed login and registration
- **District-based alerts**: Monitors your selected district with live GEE + U-Net inference
- **Risk dashboard**: Current flood %, 2010 historical baseline, composite risk score (1 to 10)
- **Satellite map**: Flood mask overlaid on Sentinel-1 radar imagery
- **FFD integration**: Live river station status when a match is found
- **Dark-themed UI**: Bottom navigation (Home, Analysis, Profile)

### Run the mobile app

```bash
cd mobile_app
pip install flet requests python-dotenv
python main.py
```

> Screenshots are stored in `public/flat mobile app/`. Requires the same GEE credentials and model weights as the web dashboard.

---

## Satellite Intelligence

FloodSense-PK uses **two complementary satellites**: one for _now_ (radar) and one for _then_ (optical historical benchmark).

---

### Sentinel-1: Synthetic Aperture Radar (The "Now")

<p align="center">
  <img src="public/Sentinel-1.png" width="420" alt="Sentinel-1 satellite" />
</p>

**Sentinel-1** is a European Space Agency mission carrying **C-band SAR** at 5.4 GHz. Unlike cameras, it actively transmits radar pulses and measures backscatter, so it operates through clouds, rain, smoke, and at night.

| Property          | Detail                                                                    |
| ----------------- | ------------------------------------------------------------------------- |
| **Revisit**       | ~6 days (constellation)                                                   |
| **Resolution**    | 10 m (IW mode)                                                            |
| **Polarizations** | VV, VH (this app uses VV from GEE; VH approximated at inference)          |
| **Flood physics** | Open water acts as a specular mirror → very weak return → **dark in SAR** |

**Why SAR wins during monsoon:**

| Capability         | Optical (Landsat / S2) | SAR (Sentinel-1)   |
| ------------------ | ---------------------- | ------------------ |
| Through clouds     | No, blocked            | Yes, penetrates    |
| Night operation    | No, needs sunlight     | Yes, active sensor |
| Peak flood capture | No, often blind        | Yes, reliable      |

**In this project:** GEE fetches `COPERNICUS/S1_GRD` VV median over the selected date window, applies an **SRTM slope mask** (&lt; 15°) to reduce terrain false positives, then feeds the tile to U-Net.

<p align="center">

|                                 Raw SAR (Charsadda)                                  |                                      AI Flood Mask                                      |
| :----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------: |
| <img src="public/District/Charsadda_Sentinel.png" width="400" alt="Charsadda SAR" /> | <img src="public/District/mask_Charsadda.png" width="400" alt="Charsadda flood mask" /> |

</p>

<p align="center"><em>District-level Sentinel-1 backscatter and U-Net flood overlay</em></p>

<p align="center">

|                                 Province-scale SAR                                 |                                Province flood heatmap                                 |
| :--------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: |
| <img src="public/Provence/Sentinel_province.png" width="400" alt="Province SAR" /> | <img src="public/Provence/heatmap_provence.png" width="400" alt="Province heatmap" /> |

</p>

---

### Landsat-5: Optical Historical Baseline (The "Then")

<p align="center">
  <img src="public/landsat-5.png" width="420" alt="Landsat 5 satellite" />
</p>

**Landsat-5** (NASA/USGS) provides multispectral optical imagery used here to reconstruct the **2010 Great Pakistan Floods**, the worst flood disaster in the country's modern history.

| Property      | Detail                                                |
| ------------- | ----------------------------------------------------- |
| **Era used**  | July–September 2010 (peak flood window)               |
| **Baseline**  | 2009 permanent water subtracted                       |
| **Method**    | **MNDWI**: Modified Normalized Difference Water Index |
| **Formula**   | `(Green − SWIR1) / (Green + SWIR1)`                   |
| **Bands**     | Landsat 5 TM C2 L2: Green `SR_B2`, SWIR1 `SR_B5`      |
| **Threshold** | MNDWI &gt; −0.1 (sensitive to turbid floodwater)      |

**Pipeline logic:** Compute 2010 flood water mask → subtract 2009 baseline water → district flood % via GEE zonal stats → compare to current SAR detection for **Delta Severity**.

<p align="center">

|                               District: 2010 flood footprint                               |                                 Province: Landsat baseline                                  |
| :----------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| <img src="public/District/Charsadda_landsat5_2010.png" width="400" alt="Charsadda 2010" /> | <img src="public/Provence/Landsat5_province.png" width="400" alt="Province Landsat 2010" /> |

|                               Province water mask                                |                                       Reference river network                                        |
| :------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
| <img src="public/Provence/bluemask.png" width="400" alt="Province water mask" /> | <img src="public/Provence/acutal_punjab_map_of_rivers.png" width="500" alt="Punjab river network" /> |

</p>

---

## Deep Learning Engine

The segmentation backbone is a **U-Net with ResNet34 encoder**, trained on the global **Sen1Floods11** benchmark and integrated into this platform via tiled GEE export inference.

### Pipeline

```mermaid
flowchart TB
    A[GEE Sentinel-1 VV GeoTIFF]
    B[Normalize VV VH ratio]
    C[Slice into 256x256 tiles]
    D[U-Net ResNet34 Sigmoid threshold 0.5]
    E[Stitch tiles flood mask and probability map]
    F[Coverage percent affected km2 risk score]
    A --> B --> C --> D --> E --> F
```

### Architecture (U-Net + ResNet34)

```mermaid
flowchart LR
    IN[GEE Sentinel-1 VV GeoTIFF]
    NORM[Normalize VV VH ratio]
    TILE[256x256 tiles]
    ENC[ImageNet encoder]
    DEC[Decoder with skip connections]
    SIG[Sigmoid threshold 0.5]
    MASK[Stitched flood mask]
    MET[Coverage km2 and risk]

    IN --> NORM --> TILE --> ENC --> DEC --> SIG --> MASK --> MET
```

### Training highlights (model repo)

| Setting      | Value                                         |
| ------------ | --------------------------------------------- |
| Loss         | Dice + Focal (γ=2)                            |
| Optimizer    | AdamW, lr=5e−5                                |
| Scheduler    | ReduceLROnPlateau (patience=5)                |
| Patch size   | 256×256, 50% overlap                          |
| Augmentation | SAR-safe geometry only (flip, rotate, affine) |

**Why only 3 SAR channels from Sen1Floods11's 8?** Optical and precipitation channels fail during cloud-covered floods. Three-channel SAR enables ImageNet transfer learning and encodes the core flood backscatter physics.

### IoU 0.5503: what it means

```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
```

A score of **0.55** on Sen1Floods11 is a solid single-model result without ensembling: honest performance on a globally diverse, peer-reviewed benchmark.

| Approach                        | Typical IoU (Sen1Floods11) |
| ------------------------------- | -------------------------- |
| dB threshold only               | ~0.30                      |
| U-Net no pretrain               | 0.40–0.50                  |
| **This model (U-Net ResNet34)** | **0.5503**                 |
| Attention U-Net                 | 0.52–0.62                  |
| Published ensembles             | up to ~0.78                |

> Place trained weights at `models/best_flood_model.pth` (not committed, large binary).

---

## Results and Visual Outputs

### District level: Charsadda case study

End-to-end outputs for a priority flood-prone district: SAR input, AI mask, probability heatmap, 2010 baseline, and matched FFD station context.

<p align="center">

|                               Sentinel-1 SAR input                               |                               U-Net flood mask                                |
| :------------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| <img src="public/District/Charsadda_Sentinel.png" width="380" alt="SAR input" /> | <img src="public/District/mask_Charsadda.png" width="380" alt="Flood mask" /> |

|                            AI probability heatmap                             |                                  2010 Landsat-5 baseline                                  |
| :---------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
| <img src="public/District/heatmap_Charsadda.png" width="380" alt="Heatmap" /> | <img src="public/District/Charsadda_landsat5_2010.png" width="380" alt="2010 baseline" /> |

</p>

<p align="center">
  <img src="public/District/waterflow_Charsadda_online.png" width="720" alt="Live FFD river status (matched station)" />
</p>

<p align="center"><em>Live FFD river status for matched Charsadda station</em></p>

### Province level: Punjab overview

Low-resolution provincial scans (~1000 m) for rapid situational awareness across large areas.

<p align="center">

|                                Sentinel-1 province scan                                 |                            Landsat-5 2010 province baseline                            |
| :-------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| <img src="public/Provence/Sentinel_province.png" width="400" alt="Province Sentinel" /> | <img src="public/Provence/Landsat5_province.png" width="400" alt="Province Landsat" /> |

|                               Provincial flood heatmap                                |                                MNDWI water mask                                |
| :-----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| <img src="public/Provence/heatmap_provence.png" width="400" alt="Province heatmap" /> | <img src="public/Provence/bluemask.png" width="400" alt="Province bluemask" /> |

</p>

---

## Historical Benchmarking (2010)

The **Overview** tab aligns 2010 Landsat MNDWI with current Sentinel-1 / U-Net detection:

| Metric                | Meaning                                            |
| --------------------- | -------------------------------------------------- |
| **2010 Historical %** | District area flooded during 2010 peak (GEE zonal) |
| **Current Flood %**   | Live SAR + U-Net detection                         |
| **Delta Severity**    | Current % − 2010 %                                 |

**Interpretation:**

- **Positive delta (+)** → Current flooding **exceeds** the 2010 disaster footprint → **CRITICAL**
- **Negative delta (−)** → Situation **safer** than the 2010 benchmark

Comparative severity is also expressed as a ratio: `current / 2010 × 100%`.

---

## Hydraulic Command Center

Real-time river intelligence scraped from Pakistan's official **FFD** (Flood Forecasting Division) portal.

### Features

- **20+ monitoring stations**: Tarbela, Sukkur, Kotri, Taunsa, Guddu, and tributary links
- **Status classification**: NORMAL, HIGH, EXTREME, NOT_RECEIVED
- **Trend detection**: Inflow / outflow Rising, Falling, Steady
- **Native Pydeck map** (web): Color-coded stations and upstream to downstream path layers
- **Analytics**: Top-10 bar charts, inflow vs outflow scatter by status

<p align="center">
  <img src="public/Provence/acutal_punjab_map_of_rivers.png" alt="Pakistan river network FFD topology" width="720" />
</p>

<p align="center"><em>Indus main stem and tributary barrage network visualized in the River Flows tab</em></p>

**River topology encoded in app:**

```
Tarbela → Besham → Kala Bagh → Chashma → Taunsa → Guddu → Sukkur → Kotri
Nowshera → Kala Bagh          (Kabul)
Mangla → Trimmu               (Jhelum)
Marala → Khanki → Qadirabad → Trimmu → Punjnad → Guddu
```

---

## Strategic AI Insights

Powered by **Gemini** (primary) with **Groq** fallback (web app only). Reports are numerically grounded: the model receives flood %, 2010 delta, risk score, and matched station hydraulic data.

**Structured output sections:**

| Section                  | Content                                      |
| ------------------------ | -------------------------------------------- |
| `[SITUATION SUMMARY]`    | Evidence-based inundation overview           |
| `[HYDRAULIC ANALYSIS]`   | Links barrage cusecs to ground flooding      |
| `[HISTORICAL BENCHMARK]` | 2010 comparison narrative                    |
| `[OPERATIONAL ACTIONS]`  | Data-backed instructions for relief agencies |
| `[CONFIDENCE]`           | Report fidelity given data gaps              |

If no API keys are configured, the engine returns a deterministic simulated briefing.

---

## Risk Scoring

**Defensible composite risk (1 to 10)**: transparent weighted formula in `engine/ai_alerts.py`:

```mermaid
pie title Risk Score Weights
    "Flood extent" : 40
    "Delta vs 2010" : 30
    "Hydraulic status" : 30
```

| Factor        | Weight | Logic                                    |
| ------------- | ------ | ---------------------------------------- |
| Flood extent  | 40%    | Scales with current inundation %         |
| Delta vs 2010 | 30%    | Spikes when today exceeds 2010 benchmark |
| River status  | 30%    | EXTREME=10, HIGH=7, NORMAL=2             |

Simple pixel risk from U-Net alone: `min(10, max(1, round(flood_pct / 10)))`.

---

## Project Structure

```
GDG-Flood-forcast/
│
├── streamlit_app.py          # Executive dashboard (primary UI)
├── main.py                   # CLI batch analysis engine
│
├── models/
│   ├── model_inference.py    # Tiled U-Net inference + metrics
│   └── best_flood_model.pth  # Weights (add locally, ~93 MB)
│
├── engine/
│   ├── ai_alerts.py          # Gemini / Groq + risk scoring
│   └── data_manager.py       # JSON export utilities
│
├── agent/                    # Disaster + Response/Communication agents
│   ├── disaster_agent.py     # Multi-stream risk assessment
│   ├── response_agent.py     # Safe-zone eval, routing, alert text
│   ├── response_schemas.py   # Pydantic models (shelters, routes, flood state)
│   ├── email_notifier.py     # Personal flood-alert email (SMTP)
│   └── mock_osm.py           # Mock OSM shelters + road graph
│
├── utils/
│   ├── ndwi.py               # Landsat-5 2010 MNDWI pipeline
│   ├── districts.py          # District boundaries + zonal stats
│   ├── visualize.py          # Static map plotting
│   └── export.py             # Output helpers
│
├── scrapers/
│   └── ffc_scraper.py        # Live FFD river discharge parser
│
├── mobile_app/               # Flet mobile alert app
│   ├── main.py
│   ├── alert_engine.py
│   └── database.py
│
├── public/                   # README screenshots & result figures
│   ├── architecture/         # Sequence diagram PNG fallback
│   ├── dashboard/            # Streamlit UI captures
│   ├── flat mobile app/      # Flet mobile app screenshots
│   ├── District/             # Charsadda case-study outputs
│   ├── Provence/             # Province-level outputs
│   ├── Sentinel-1.png
│   └── landsat-5.png
│
├── pakistan_districts.json   # District boundary GeoJSON
├── outputs/                  # Generated maps (runtime)
├── requirements.txt
└── readme.md
```

---

Here is the complete, integrated **Installation and GEE Setup** section for your `README.md`. It seamlessly combines your installation steps with the detailed Earth Engine onboarding guide into a single, cohesive workflow.

---

## 🛠️ Installation & Configuration

### Prerequisites

- Python **3.10+**
- [Google Earth Engine](https://earthengine.google.com/) account
- (Optional) NVIDIA GPU for faster model inference (CPU fallback supported)
- API keys: `GEMINI_API_KEY` and/or `GROQ_API_KEY` for AI automated reporting

---

### Installation

```bash
# Clone the repository
git clone https://github.com/HAMZOO0/FloodSense-PK-RealTime-Flood-Detection-Sentinel1-UNet.git
cd FloodSense-PK-RealTime-Flood-Detection-Sentinel1-UNet

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS

# Install core dependencies
pip install -r requirements.txt

```

---

### 2. Google Earth Engine (GEE) Services Setup

This application utilizes **Google Earth Engine** to pull and process real-time geospatial imagery (such as Sentinel-1 SAR data). To handle automated background API calls safely, you must configure a Google Cloud Service Account.

#### A. Project Registration

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Click the project dropdown in the top navigation bar, select **New Project**, and copy down your unique **Project ID**.
3. Enable the API: Navigate to the [Google API Library](https://console.cloud.google.com/apis/library), search for **Google Earth Engine API**, and click **Enable**.
4. **Mandatory Step:** Visit the [Earth Engine Project Registration page](https://console.cloud.google.com/earth-engine/configuration) and register your project under the **Noncommercial / Academic / Research** tier (or your preferred active plan).

#### B. Service Account Identity & Permissions

1. Go to **IAM & Admin** > [Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
2. Click **+ Create Service Account**, assign it a name (e.g., `floodsense-backend`), and click **Create and Continue**.
3. Go to **IAM & Admin** > **IAM** and click **+ Grant Access**. Add your new service account email as a **New Principal** and attach these two mandatory roles:

- **Earth Engine Resource Viewer** (Allows data reads/queries)
- **Service Usage Consumer** (Allows your project to use account quotas)

4. Go back to the **Service Accounts** page, click your account, navigate to the **Keys** tab, and select **Add Key** > **Create new key** > **JSON**. This downloads a private credentials file to your machine.

---

### 3. Application Secrets Management

The app uses two explicit mechanisms to store sensitive API credentials safely out of version control. Ensure your local workspace files match this layout:

#### A. Streamlit Native Config (`secrets.toml`)

Streamlit securely passes Earth Engine credentials natively through a hidden configuration directory.

1. Create the local structure inside the project root:

```bash
mkdir .streamlit

```

2. Open your downloaded service account JSON file, and translate its values into `.streamlit/secrets.toml`:

```toml
[gee]
type = "service_account"
project_id = "floodsense-pk"
private_key_id = "YOUR_DOWNLOADED_PRIVATE_KEY_ID"
private_key = '''-----BEGIN PRIVATE KEY-----
YOUR_MASSIVE_MULTILINE_KEY_HERE
-----END PRIVATE KEY-----'''
client_email = "your-service-account@floodsense-pk.iam.gserviceaccount.com"
client_id = "YOUR_DOWNLOADED_CLIENT_ID"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "YOUR_CLIENT_X509_CERT_URL"
universe_domain = "googleapis.com"

```

> ⚠️ **Formatting Note:** Use **triple single-quotes (`'''`)** for the `private_key` parameter as shown above to allow TOML to parse the multi-line block string cleanly without runtime line-break parsing errors.

#### B. LLM Provider Config (`.env`)

Create a standard `.env` file in the root directory to store variables for report generation models:

```env
PROJECT_ID="your-google-cloud-project-id"
GEMINI_API_KEY="your-gemini-key"
GROQ_API_KEY="your-groq-key"
EMAIL_SENDER="YOUR_EMAIL_SENDER"
EMAIL_APP_PASSWORD="your 16 char app password"
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"

```

> The `GEMINI_API_KEY` / `GROQ_API_KEY` power the LLM report generation. The four `EMAIL_*` variables power the **Personal Flood Alert** emails (see the next section). All of them are optional — the app runs without them; only the corresponding feature is disabled when a key is absent.

#### B-1. Personal Flood Alert Email (Gmail SMTP)

The **Response & Communication Agent** can email a personalised evacuation alert to a citizen when their selected area is in danger. Delivery uses plain SMTP, so any provider works, but the simplest zero-cost option is a **Gmail App Password**:

| Variable | What it is | Example |
| --- | --- | --- |
| `EMAIL_SENDER` | The Gmail address the alert is sent **from** | `floodsense.alerts@gmail.com` |
| `EMAIL_APP_PASSWORD` | A 16-character Google App Password (**not** your normal login password) | `abcd efgh ijkl mnop` |
| `SMTP_HOST` | SMTP server host (defaults to Gmail) | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP STARTTLS port | `587` |

**Generate a Gmail App Password (≈2 minutes):**

1. Use a Google account and enable **2-Step Verification** at <https://myaccount.google.com/security>.
2. Open <https://myaccount.google.com/apppasswords>.
3. Create a password for the app **"Mail"** — Google shows a 16-character code.
4. Paste that code into `EMAIL_APP_PASSWORD` (spaces are fine) and set `EMAIL_SENDER` to that Gmail address.

> **Non-Gmail providers:** set `SMTP_HOST` / `SMTP_PORT` to your provider's STARTTLS endpoint and use the corresponding SMTP username (`EMAIL_SENDER`) and password (`EMAIL_APP_PASSWORD`). No code changes are needed.

Once configured, the agent sends each alert with the **current situation** of the affected area (risk level, flood coverage, affected area, estimated population at risk, available shelters), the **recommended safe zone with its exact coordinates** and a Google Maps directions link, and the **distance and estimated travel time** to reach it.

#### C. Safety Guardrails

Ensure your `.gitignore` contains the following lines so your production access keys are never pushed to GitHub:

```text
.env
.streamlit/secrets.toml

```

---

### 4. Running the Application

Before initializing execution, place your trained UNet deep learning framework checkpoint file into the proper local relative workspace directory path:

```text
models/best_flood_model.pth

```

Once the weights are verified and keys are loaded, kickstart the interactive visualization server:

```bash
streamlit run app.py

```

## Usage

### Launch the web dashboard

```bash
streamlit run streamlit_app.py
```

1. Select **District / Province / National** scale (sidebar).
2. Pick date range for current SAR composite.
3. *(Optional)* Under **🔔 Personal Flood Alert**, enter **Your email** to be notified if the area is in danger. By default the alert is emailed only when the risk is **HIGH**; tick **"Email me regardless of risk level"** to receive it at any risk level (useful for testing). Requires the `EMAIL_*` keys from the [setup section](#b-1-personal-flood-alert-email-gmail-smtp).
4. Click **Run analysis**.
5. Explore tabs: Overview → Detection → River Flows → AI Intelligence.

If you subscribed an email, a confirmation banner appears after the run showing the recommended safe zone, its coordinates, distance, and estimated travel time — and the alert lands in the recipient's inbox.

Or use the hosted version: **[floodsense-pk.streamlit.app](https://floodsense-pk.streamlit.app/)**

### Launch the mobile app

```bash
cd mobile_app
python main.py
```

1. **Register**: pick your district from 148 options.
2. **Login**: access your personalized dashboard.
3. Navigate **Home, Analysis, Profile** via the bottom bar.

### CLI batch mode

```bash
python main.py                    # default quick districts
python main.py --district Larkana # single district
```

### Inference API (programmatic)

```python
from models.model_inference import load_flood_model, predict_flood

model = load_flood_model()
result = predict_flood(model, sar_geotiff_bytes, "Charsadda", bbox)

print(result["water_coverage_pct"], result["affected_area_km2"], result["risk_score"])
```

---

## Limitations

| Limitation               | Impact                                 | Mitigation in FloodSense-PK                                   |
| ------------------------ | -------------------------------------- | ------------------------------------------------------------- |
| VH approximated from VV  | Slightly weaker VH/VV ratio            | Full 8-ch Sen1Floods11 training; future dual-pol GEE export   |
| Flooded vegetation       | SAR underestimation possible           | Ratio channel + slope mask                                    |
| Urban flooding           | Shadowed water invisible to SAR        | Combine with DEM / optical when clear                         |
| Small floods             | May miss at coarse national resolution | Use District scale (~80 m)                                    |
| Domain shift             | Global Sen1Floods11 → Pakistan         | Priority districts; local fine-tuning planned                 |
| 2010 optical vs 2024 SAR | Different sensors / methods            | Delta used as **benchmark indicator**, not pixel-perfect diff |
| FFD scraper fragility    | Site layout changes break parser       | Regex-based aggressive parsing + status fallbacks             |
| Temporal lag             | Reflects last SAR overpass             | Combine with live FFD discharge                               |

---

## Project Tools & Dependencies

All libraries and services used across the web dashboard, CLI pipeline, and mobile app. Full list in [`requirements.txt`](requirements.txt).

### Core Python

|                                                                                                                                             | Library             | Purpose                                  |
| :-----------------------------------------------------------------------------------------------------------------------------------------: | ------------------- | ---------------------------------------- |
|        <img src="https://img.shields.io/badge/-numpy-013243?style=flat-square&logo=numpy&logoColor=white" height="18" alt="numpy"/>         | **numpy**           | Numerical computing and array operations |
|       <img src="https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white" height="18" alt="pandas"/>       | **pandas**          | Data manipulation and analysis           |
| <img src="https://img.shields.io/badge/-matplotlib-11557c?style=flat-square&logo=matplotlib&logoColor=white" height="18" alt="matplotlib"/> | **matplotlib**      | Data visualization and plotting          |
|       <img src="https://img.shields.io/badge/-Pillow-FFE873?style=flat-square&logo=python&logoColor=black" height="18" alt="pillow"/>       | **pillow**          | Image loading and basic processing       |
|      <img src="https://img.shields.io/badge/-requests-010101?style=flat-square&logo=curl&logoColor=white" height="18" alt="requests"/>      | **requests**        | HTTP requests and API communication      |
|                                                                                                                                             | **python-dotenv**   | Environment variable management          |
|                                                                                                                                             | **python-dateutil** | Date and time parsing utilities          |

### Web Scraping (FFD portal)

|     | Library                                           | Purpose                                  |
| :-: | ------------------------------------------------- | ---------------------------------------- |
|     | **beautifulsoup4 / bs4**                          | HTML parsing for FFD river-state scraper |
|     | **soupsieve**                                     | CSS selector support for BeautifulSoup   |
|     | **urllib3 · certifi · charset-normalizer · idna** | HTTP client, SSL, and encoding utilities |

### Geospatial & Remote Sensing

|                                                                                                                                                        | Library / Service   | Purpose                                       |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------- | --------------------------------------------- |
| <img src="https://img.shields.io/badge/-Google_Earth_Engine-4285F4?style=flat-square&logo=google-earth-engine&logoColor=white" height="18" alt="GEE"/> | **earthengine-api** | Satellite imagery analysis at planetary scale |
|                                                                                                                                                        | **rasterio**        | Read / write geospatial raster data           |
|                                                                                                                                                        | **geopandas**       | Geospatial dataframes and vector data         |
|                                                                                                                                                        | **shapely**         | Geometric operations on vector shapes         |
|            <img src="https://img.shields.io/badge/-pydeck-000000?style=flat-square&logo=mapbox&logoColor=white" height="18" alt="pydeck"/>             | **pydeck**          | Large-scale geospatial map visualization      |

### Computer Vision & Deep Learning

|                                                                                                                                    | Library                         | Purpose                                              |
| :--------------------------------------------------------------------------------------------------------------------------------: | ------------------------------- | ---------------------------------------------------- |
| <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" height="18" alt="pytorch"/> | **torch · torchvision**         | Deep learning framework and image transforms         |
|                                                                                                                                    | **opencv-python-headless**      | Image processing and computer vision                 |
|                                                                                                                                    | **albumentations**              | SAR-safe image augmentation for training             |
|                                                                                                                                    | **segmentation-models-pytorch** | Pretrained U-Net / ResNet segmentation architectures |

### Web Application & UI

|                                                                                                                                          | Library       | Purpose                                                                       |
| :--------------------------------------------------------------------------------------------------------------------------------------: | ------------- | ----------------------------------------------------------------------------- |
| <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" height="18" alt="streamlit"/> | **streamlit** | Interactive web dashboard ([live demo](https://floodsense-pk.streamlit.app/)) |
|       <img src="https://img.shields.io/badge/-Flet-02569B?style=flat-square&logo=flutter&logoColor=white" height="18" alt="flet"/>       | **flet**      | Cross-platform mobile alert app (`mobile_app/`)                               |

### Google Cloud & APIs

|                                                                                                                                          | Library / Service                          | Purpose                               |
| :--------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------ | ------------------------------------- |
| <img src="https://img.shields.io/badge/-Google_Cloud-4285F4?style=flat-square&logo=google-cloud&logoColor=white" height="18" alt="gcp"/> | **google-cloud-storage**                   | Cloud Storage access                  |
|                                                                                                                                          | **google-api-python-client · google-auth** | Google REST APIs and authentication   |
|                                                                                                                                          | **googleapis-common-protos · proto-plus**  | Protocol Buffer types for Google APIs |

### AI & LLM Services

|                                                                                                                                        | Library / Service                | Purpose                            |
| :------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------- | ---------------------------------- |
| <img src="https://img.shields.io/badge/-Gemini-8E75B2?style=flat-square&logo=google-gemini&logoColor=white" height="18" alt="gemini"/> | **google-generativeai · Gemini** | Primary tactical report generation |
|      <img src="https://img.shields.io/badge/-Groq-F55036?style=flat-square&logo=fastapi&logoColor=white" height="18" alt="groq"/>      | **groq**                         | Fast LLM inference fallback        |

### Satellites & Data Sources

|                                                                                                                                             | Source               | Role in FloodSense-PK                   |
| :-----------------------------------------------------------------------------------------------------------------------------------------: | -------------------- | --------------------------------------- |
| <img src="https://img.shields.io/badge/-Sentinel--1-0078D4?style=flat-square&logo=satellite&logoColor=white" height="18" alt="sentinel-1"/> | **Sentinel-1 SAR**   | Live flood detection through clouds     |
|    <img src="https://img.shields.io/badge/-Landsat--5-2E7D32?style=flat-square&logo=nasa&logoColor=white" height="18" alt="landsat-5"/>     | **Landsat-5 (2010)** | Historical MNDWI flood benchmark        |
|                                                                                                                                             | **FFD River Portal** | Live barrage discharge and river status |

### Security & Build Utilities

|     | Library                                                                 | Purpose                                     |
| :-: | ----------------------------------------------------------------------- | ------------------------------------------- |
|     | **cryptography · cffi · pycparser**                                     | Cryptographic primitives (GEE / HTTPS deps) |
|     | **packaging · fonttools · kiwisolver · contourpy · cycler · pyparsing** | Matplotlib and packaging support            |

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
2. He, K., et al. (2016). **Deep Residual Learning for Image Recognition.** [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
3. Lin, T. Y., et al. (2017). **Focal Loss for Dense Object Detection.** [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
4. Bonafilia, D., et al. (2020). **Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1.** CVPR EarthVision Workshop.
5. Torres, R., et al. (2012). **GMES Sentinel-1 Mission.** _Remote Sensing of Environment_, 120, 9–24.
6. Xu, H. (2006). **Modification of Normalised Difference Water Index (MNDWI).** _Int. J. Remote Sensing_, 27(14).
7. Pakistan Flood Forecasting Division: [River State Portal](https://ffd.pmd.gov.pk/river-state)

---

<div align="center">

**Developed for the GDG Flood Forecast Challenge 2026**

_FloodSense-PK: See through the clouds. Compare to 2010. Act on live river data._

[![Live Demo](https://img.shields.io/badge/Try_Live_Demo-floodsense--pk.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://floodsense-pk.streamlit.app/)

</div>
