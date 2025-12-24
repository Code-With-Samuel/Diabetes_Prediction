# Diabetes Risk Prediction 🏥

**A compact, explainable ML project that predicts an individual's risk of developing diabetes based on health metrics.** The project includes an exploratory notebook, a trained LightGBM model, and a Streamlit app for interactive predictions.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model & Features](#model--features)
- [How the App Works](#how-the-app-works)
- [Retraining / Experiments](#retraining--experiments)
- [Deployment & Usage](#deployment--usage)
- [Notes & Disclaimer](#notes--disclaimer)
- [License & Contact](#license--contact)

---

## Overview ✅

This repository contains a machine learning pipeline and demo application to predict diabetes risk from structured health and lifestyle features. The project was developed and evaluated in `diabetes_prediction.ipynb` and the best-performing model (LightGBM) is saved as `diabetes_model_LightGBM.pkl` for inference inside a Streamlit app (`app.py`).

---

## Quick Start 🚀

1. Create an environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Run the Streamlit app locally:

```bash
streamlit run app.py
```

3. Open the displayed URL in your browser, fill in health inputs, and click **Predict Diabetes Risk**.

---

## Project Structure 🔧

- `app.py` — Streamlit app for interactive predictions and risk explanations.
- `diabetes_prediction.ipynb` — Notebook with EDA, feature engineering, model training and evaluation.
- `diabetes_model_LightGBM.pkl` — Trained LightGBM classifier used by `app.py` (model file).
- `feature_names.json` — Ordered list of features expected by the model.
- `model_metadata.pkl` — Saved metadata about the model/training (if present).
- `submission.csv` — Example output/predictions on the test partition.
- `requirements.txt` — Python dependencies used by the project.

---

## Model & Features 📊

- **Model:** LightGBM classifier (saved as `diabetes_model_LightGBM.pkl`)
- **Reported performance:** Best AUC ≈ **0.7148** (3-fold CV, see `diabetes_prediction.ipynb`)
- **Feature list:** See `feature_names.json`. Important derived features include:
  - `bmi_age` — interaction: BMI × age
  - `waist_bmi` — waist-to-hip ratio × BMI
  - `pulse_pressure` — systolic − diastolic
  - `chol_hdl_ratio`, `trig_hdl_ratio` — lipid ratios
  - `health_score` — composite lifestyle score (diet, activity, sleep, screen-time)
  - `is_senior`, `is_obese` — binary flags

These features are created in `create_features()` inside `app.py` and in the notebook during preprocessing.

---

## How the App Works 🧠

- The app loads `diabetes_model_LightGBM.pkl` using `joblib`.
- User inputs (demographics, vitals, labs, lifestyle) are converted into a single-row DataFrame in the exact order expected by the model, engineered via `create_features()`, and then passed to `model.predict_proba()`.
- The UI shows the predicted probability and a **risk level** category derived from thresholds in the app:
  - **Low Risk**: probability < 0.3
  - **Moderate Risk**: 0.3 ≤ probability < 0.5
  - **High Risk**: 0.5 ≤ probability < 0.7
  - **Very High Risk**: probability ≥ 0.7

> ⚠️ **Important:** This is a demonstrative model for educational/informational purposes only. It is not a substitute for professional medical diagnosis.

---

## Retraining & Experiments 🧪

- All training, hyperparameter evaluation (multiple algorithms were compared), and model selection are performed in `diabetes_prediction.ipynb`. The notebook uses scikit-learn pipelines and evaluates models using ROC AUC with cross-validation.
- To retrain: open the notebook, adjust preprocessing or model hyperparameters, re-run the training cells and save the best model. The notebook contains cells that save:
  - `diabetes_model_LightGBM.pkl`
  - `feature_names.json`
  - `model_metadata.pkl`

---

## Deployment & Usage Tips ⚙️

- For local demo: `streamlit run app.py` is sufficient.
- To deploy: host the repository on a VM or use Streamlit Cloud / other platforms that support Streamlit apps. Ensure `diabetes_model_LightGBM.pkl` is present in the app directory.
- Monitor inputs closely: the model expects numerical ranges similar to typical adult health metrics; out-of-distribution inputs may produce unreliable probabilities.

---

## Notes & Disclaimer ✍️

- **Data privacy:** No personal data is stored by this demo app; it runs locally and predictions are ephemeral.
- **Medical disclaimer:** The predictions are probabilistic estimates from an ML model and should not be used for clinical decisions.

---

## License & Contact 📨

If you use or adapt this project, please add proper attribution. For questions or contributions, open an issue or contact the maintainer.

**Enjoy exploring the model!**
