# ExoVision-AI 

AI-powered pipeline for **automatic exoplanet detection** using NASA’s **Kepler** open-source dataset.  
Built during the **NASA Space Apps Hackathon 2025** by a team of 5 developers.

---

## Project Overview
Thousands of exoplanets have been identified from Kepler’s transit data, yet much of this work was done manually.  
**ExoVision-AI** uses modern Machine Learning to classify light curve data points as:

- ✅ Confirmed Exoplanet  
- 🟡 Planetary Candidate  
- ❌ False Positive  

Our goal is to provide a reproducible model and a simple web interface where scientists and enthusiasts can upload new Kepler data and instantly receive classification results.

---

## Key Features
- **Kepler dataset integration** – cleaned & preprocessed for ML.
- **Automated classification** – using gradient boosting & deep learning models.
- **Interactive web interface** – upload new light curves and view predictions.
- **Metrics dashboard** – model accuracy, ROC curves, and confusion matrices.
- **Hackathon-friendly** – quick setup and reproducible environment.

---

## Dataset
- **Source:** [NASA Exoplanet Archive – Kepler Mission](https://exoplanetarchive.ipac.caltech.edu/)  
- **Variables used:** Orbital period, transit duration, planetary radius, stellar parameters, etc.

---

## Model Architecture
- Preprocessing pipeline for missing values & scaling.
- Gradient Boosting (LightGBM/XGBoost/CatBoost) experiments.
- Final ensemble model chosen for best ROC-AUC.
- Deployed behind a lightweight Python/Flask (or FastAPI) web server.

---

## Installation

```bash
# clone repo
git clone https://github.com/c0llectorr/ExoVision-AI.git
cd ExoVision-AI

# create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt
```
