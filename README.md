---
title: Face Direction Classifier
emoji: 🧭
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: latest
app_file: app.py
pinned: false
license: Apache-2.0
---

# Face Direction Classifier 🧭

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org) [![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://python.org)

An interactive **Streamlit web application** that classifies face direction (Front, Side, Back) using a fine‑tuned ResNet‑18 deep learning model.

---

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/aveline/face-direction-classifier.git
cd face-direction-classifier

# Install dependencies
pip install -r requirements.txt

# Run locally with Streamlit
streamlit run app.py
