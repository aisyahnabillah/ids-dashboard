---
title: IDS - XAI Dashboard
emoji: 🛡️
colorFrom: gray
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
---

# 🛡️ Intrusion Detection System with Explainable AI

An interactive, machine-learning-based Intrusion Detection System (IDS) dashboard built with **Streamlit**, using the **CIC-IDS-2017** dataset. The application classifies network traffic with a trained **LightGBM** model, explains model predictions with **SHAP**, and turns the technical findings into a natural-language explanation using **GPT-4o-mini**.

🔗 **Live demo:** [IDS – XAI Dashboard](https://huggingface.co/spaces/aisyahn/ids-dashboard)

## Overview

Traditional machine-learning IDS models can be difficult to interpret because their predictions do not always explain why a traffic sample is classified as benign or malicious. This project combines model-based intrusion detection with Explainable AI (XAI) and a Large Language Model (LLM) to make individual predictions easier to understand.

The dashboard provides an interactive workflow for entering network-traffic feature values, viewing the predicted class and confidence, inspecting feature contributions, and reading a contextual explanation in Indonesian.

## Key Features

- **Multi-class traffic classification** using a trained LightGBM model.
- **Quick Presets** for Normal Traffic (BENIGN), DoS Hulk, PortScan, and Bot examples, alongside a Custom option.
- **Manual feature input** for key network-traffic attributes.
- **Advanced Input** to select and override additional model features.
- **Prediction results** with the predicted class, confidence score, severity label, and top class probabilities.
- **SHAP-based explainability** with a feature-contribution chart and a list of the top contributing features.
- **LLM-generated explanation** using GPT-4o-mini to describe the prediction, interpret relevant traffic indicators, and suggest a response when an attack is detected.

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web application | Streamlit |
| Main classification model | LightGBM |
| Model explainability | SHAP TreeExplainer |
| LLM explanation | OpenAI GPT-4o-mini |
| Data processing | pandas, NumPy |
| Model/artifact loading | joblib |
| Visualization | Matplotlib |
| Dataset | CIC-IDS-2017 |

## Model Information

The deployed dashboard uses **LightGBM** as its primary classifier. The repository also contains Random Forest and XGBoost model artifacts for the model-comparison work.

The dashboard displays the following model metrics in its sidebar:

| Metric | Value |
|---|---:|
| Accuracy | 99.8% |
| Macro Precision | 0.878 |
| Macro Recall | 0.953 |
| Macro F1-score | 0.905 |

These are the values currently displayed by the application. Refer to the accompanying evaluation artifacts and notebooks for the model-selection and analysis workflow.

## Repository Structure

```text
ids-dashboard/
├── app.py
├── requirements.txt
├── model_results.csv
├── predictions.pkl
├── results.pkl
├── data/
│   ├── feature_columns.json
│   ├── preset_samples.json
│   └── shap_per_class.json
├── models/
│   ├── LightGBM.pkl
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│   ├── label_encoder.pkl
│   └── shap_explainer.pkl
└── notebook/
    ├── model_selection.ipynb
    └── shap_llm.ipynb
```

- `app.py` — Streamlit application, prediction pipeline, SHAP visualization, and LLM integration.
- `data/` — feature-column definitions, preset traffic samples, and SHAP information.
- `models/` — trained model artifacts, label encoder, and SHAP explainer.
- `notebook/` — notebooks for model selection and SHAP/LLM analysis.
- `model_results.csv`, `predictions.pkl`, and `results.pkl` — saved model evaluation and prediction-related artifacts.

## Run Locally

### 1. Clone the repository

```bash
git clone https://huggingface.co/spaces/aisyahn/ids-dashboard
cd ids-dashboard
```

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the OpenAI API key

The application uses the `OPENAI_API_KEY` environment variable for GPT-4o-mini explanations. Set your key locally before starting the app.

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

**macOS / Linux:**

```bash
export OPENAI_API_KEY="your-api-key"
```

Do not commit API keys or other credentials to the repository. For Hugging Face Spaces, add `OPENAI_API_KEY` under the Space's **Settings → Variables and secrets → Secrets**.

If the API key is not configured, the IDS prediction and SHAP workflow can still run, but the LLM explanation will not be available.

### 5. Start the dashboard

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, usually `http://localhost:8501`.

## How to Use the Dashboard

1. Open the [live dashboard](https://huggingface.co/spaces/aisyahn/ids-dashboard).
2. Select a Quick Preset, or choose **Custom** to enter your own traffic-feature values.
3. Adjust the displayed network-traffic features.
4. Optionally expand **Advanced Input** to select and edit additional features.
5. Click **Analyze Traffic**.
6. Review the predicted traffic class, confidence score, severity, and class probabilities.
7. Inspect the SHAP chart and top contributing features.
8. Read the generated natural-language explanation and suggested action, when the OpenAI API key is configured.

## Explainability Workflow

The dashboard combines two complementary explanation layers:

1. **SHAP** estimates the contribution of input features to the model's prediction and presents the most influential features.
2. **GPT-4o-mini** receives the predicted class, confidence score, top SHAP features, and selected traffic values to generate a short Indonesian explanation with contextual interpretation and a suggested action.

The LLM is used as an interpretation layer; the traffic classification itself is produced by the trained LightGBM model.

## Dataset

This project uses **CIC-IDS-2017**, a network-intrusion detection dataset containing benign traffic and multiple attack categories. The dataset is used for training and evaluating the machine-learning models.

Please refer to the dataset's original publication and distribution terms when reusing the data. This repository contains model and application artifacts; it does not include the full raw CIC-IDS-2017 dataset.

## Notes and Limitations

- This is an academic/research prototype and should not be treated as a standalone production security system.
- Predictions depend on the quality, scale, and meaning of the input features. Inputs should follow the feature definitions and preprocessing expected by the trained model.
- The LLM-generated narrative is an additional interpretation and may be incomplete or inaccurate. Verify security decisions with appropriate network evidence and security expertise.
- The application accepts manually entered feature values and preset examples; it is not currently a live packet-capture or continuous network-monitoring system.
- Keep the OpenAI API key private and monitor API usage.

## Author

**Aisyah Nabillah**  
Computer Science, BINUS Online – Universitas Bina Nusantara
---

*Developed as an academic project exploring the integration of machine learning, Explainable AI, and Large Language Models for network intrusion detection.*
