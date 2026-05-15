# ECG Classification using Deep Learning

A deep learning-based ECG classification system built using PyTorch and the PTB-XL clinical dataset. The project classifies 12-lead ECG recordings into multiple cardiac diagnostic categories and provides real-time inference through a Streamlit interface.

## Features

- Multi-class ECG classification using a 1D CNN
- Trained on the PTB-XL clinical dataset
- Handles class imbalance using weighted loss functions
- Real-time ECG inference through Streamlit
- End-to-end preprocessing and training pipeline
- Model evaluation using macro-F1 and accuracy metrics

## Tech Stack

### Machine Learning
- Python
- PyTorch
- scikit-learn
- NumPy
- Pandas

### Deployment
- Streamlit

## Dataset

- PTB-XL ECG Dataset
- Approximately 17,000 raw 12-lead ECG recordings

## Model Architecture

- 1D Convolutional Neural Network (CNN)
- Multi-class classification for:
  - NORM
  - MI
  - STTC
  - CD

## Installation

### Clone the repository

```bash
git clone https://github.com/Arvind172/<your-repo-name>.git
cd <your-repo-name>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Project Workflow

1. ECG data preprocessing
2. Signal normalization and transformation
3. Model training using PyTorch
4. Performance evaluation
5. Real-time inference using Streamlit

## Future Improvements

- Transformer-based ECG models
- Multi-label classification
- Improved signal denoising
- Cloud deployment
- Explainable AI visualizations

## Author

Arvind S  
GitHub: https://github.com/Arvind172
