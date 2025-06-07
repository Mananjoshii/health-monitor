import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import pywt
import time

st.set_page_config(page_title="Live Health Monitor", layout="wide")
st.title("💓 Live Health Monitor (Deployed)")

# ✅ URL of deployed Flask server
API_URL = "https://health-monitor-7lno.onrender.com"  # Replace with your Flask endpoint

# --- 1. Fetch IR/RED values from deployed Flask server ---
def fetch_data():
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()['data']
            ir, red = zip(*data)
            return np.array(ir), np.array(red)
        else:
            st.error("Failed to fetch data from server.")
            return None, None
    except Exception as e:
        st.error(f"Exception occurred: {e}")
        return None, None

ir_values, red_values = fetch_data()

if ir_values is None or len(ir_values) < 100:
    st.warning("Waiting for sufficient data (600+ IR > 100000 readings)...")
    st.stop()

# --- 2. Signal Filtering / Enhancement ---
def bandpass_filter(signal, lowcut=0.1, highcut=0.8, fs=20, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def wavelet_denoise(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, np.std(c) / 2, 'soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

# --- 3. Process IR signal ---
filtered_ir = wavelet_denoise(ir_values)
filtered_ir = bandpass_filter(filtered_ir)

# --- 4. Peak Detection ---
def detect_peaks(signal, fs=20):
    peaks, _ = find_peaks(signal, distance=fs * 2, prominence=0.2 * np.max(signal))
    return peaks

peaks = detect_peaks(filtered_ir)

# --- 5. Estimation Metrics ---
duration_sec = len(filtered_ir) / 20
resp_rate = (len(peaks) / duration_sec) * 60 if duration_sec > 0 else 0
heart_rate = 84  # Dummy for now, override with real calc if you have it

# --- SpO2 Estimation (optional improvement area) ---
def estimate_spo2(ir, red, peak_indices):
    ir_ac = np.std([ir[i] for i in peak_indices])
    red_ac = np.std([red[i] for i in peak_indices])
    if red_ac == 0:
        return 0
    ratio = (ir_ac / red_ac)
    spo2 = 110 - 25 * ratio  # Empirical formula
    return np.clip(spo2, 0, 100)

spo2 = estimate_spo2(ir_values, red_values, peaks)

# --- 6. Display Results ---
st.subheader("📡 Live Signal Visualization")

st.line_chart(filtered_ir, height=250, use_container_width=True)
st.write(f"✅ Detected {len(peaks)} breaths in {int(duration_sec)} sec → Estimated RR: **{resp_rate:.2f} bpm**")

st.subheader("❤️ Heart Rate Estimation")
st.metric("Estimated Heart Rate", f"{heart_rate} BPM")

st.subheader("🩸 SpO₂ Estimation")
st.metric("Estimated SpO₂", f"{spo2:.2f} %")

st.caption("Data streamed from ESP32 via Flask server")
