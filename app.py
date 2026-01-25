import streamlit as st
import torch
import numpy as np
import pickle
import wfdb
import os
import tempfile

# ===================== MODEL =====================
import torch.nn as nn
import torch.nn.functional as F

class ECGCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * 250, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    with open("label_classes.pkl", "rb") as f:
        classes = pickle.load(f)

    model = ECGCNN(num_classes=len(classes))
    model.load_state_dict(torch.load("ecg_cnn.pt", map_location="cpu"))
    model.eval()
    return model, classes

model, classes = load_model()

# ===================== UI =====================
st.title("ECG Classification Demo")
st.write("Upload a PTB-XL ECG record (.hea and .dat files)")

st.markdown("### Upload ECG files")

hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

if hea_file and dat_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        hea_path = os.path.join(tmpdir, hea_file.name)
        dat_path = os.path.join(tmpdir, dat_file.name)

        with open(hea_path, "wb") as f:
            f.write(hea_file.read())

        with open(dat_path, "wb") as f:
            f.write(dat_file.read())

        record_name = hea_path.replace(".hea", "")

        if st.button("Run Inference"):
            signal, _ = wfdb.rdsamp(record_name, sampto=1000)
            signal = signal.T  # (12, 1000)

            x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            pred_idx = int(np.argmax(probs))

            st.success(f"Predicted Class: **{classes[pred_idx]}**")

            st.markdown("### Class Probabilities")
            for cls, p in zip(classes, probs):
                st.write(f"{cls}: {p:.3f}")
