---
title: IDS - XAI Dashboard
emoji: 🛡️
colorFrom: gray
colorTo: yellow
sdk: streamlit
sdk_version: 0.4.0
app_file: app.py
pinned: false
---

# IDS - XAI Dashboard

**Intrusion Detection System** berbasis LightGBM dengan Explainable AI (SHAP) dan narasi LLM.

## Stack
- **Model**: LightGBM, ditraining pada CIC-IDS-2017
- **XAI**: SHAP TreeExplainer
- **LLM**: GPT-4o-mini (OpenAI)
- **Framework**: Streamlit

## Struktur File

```
├── app.py                     
├── requirements.txt 
├── models/
│   ├── LightGBM.pkl
│   ├── label_encoder.pkl
│   └── shap_explainer.pkl
└── data/
    ├── feature_columns.json
    ├── preset_samples.json
    └── shap_per_class.json
```