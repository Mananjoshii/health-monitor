# ✅ Updated Streamlit Code for Deployment with Cloud ESP32 + Flask Setup

import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from collections import deque
import requests
import math

# === CONFIG ===
READ_DURATION = 30  # seconds
TOUCH_THRESHOLD = 100000
SAVE_IR_PATH = 'ir_cleaned_data.csv'
SAVE_RED_PATH = 'red_cleaned_data.csv'
API_URL = "https://health-monitor-7lno.onrender.com/latest"  # Flask server

# === ThingSpeak ===
def send_to_thingspeak(spo2, rr, hr, name, age, gender, api_key):
    url = "https://api.thingspeak.com/update"
    payload = {
        "api_key": api_key,
        "field1": rr,
        "field2": spo2,
        "field3": hr,
        "field4": age,
        "field5": gender,
        "field6": name,
    }
    try:
        response = requests.get(url, params=payload)
        if response.status_code == 200:
            st.success("📡 Data pushed to ThingSpeak!")
        else:
            st.error(f"❌ ThingSpeak push failed. Status: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error sending to ThingSpeak: {e}")

# === FILTERING ===
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def preprocess_signal(signal, fs):
    bandpassed = bandpass_filter(signal, 0.1, 0.5, fs)
    smoothed = savgol_filter(bandpassed, window_length=51, polyorder=3)
    smoothed += np.mean(signal)
    return smoothed

def detect_breath_peaks(signal, fs):
    min_interval = int(1.5 * fs)
    peaks, _ = find_peaks(signal, distance=min_interval, prominence=0.05)
    return peaks

def calculate_spo2(ir_raw, red_raw, fs):
    ir_filtered = bandpass_filter(ir_raw, 0.5, 3.0, fs)
    red_filtered = bandpass_filter(red_raw, 0.5, 3.0, fs)
    peaks, _ = find_peaks(ir_filtered, distance=int(0.6 * fs), prominence=0.02 * max(ir_filtered))
    R_values = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end - start < 3:
            continue
        ir_seg = ir_raw[start:end]
        red_seg = red_raw[start:end]
        ir_filt_seg = ir_filtered[start:end]
        red_filt_seg = red_filtered[start:end]
        AC_ir = np.max(ir_filt_seg) - np.min(ir_filt_seg)
        DC_ir = np.mean(ir_seg)
        AC_red = np.max(red_filt_seg) - np.min(red_filt_seg)
        DC_red = np.mean(red_seg)
        if DC_ir == 0 or DC_red == 0:
            continue
        R = (AC_red / DC_red) / (AC_ir / DC_ir)
        R_values.append(R)
    if not R_values:
        return 0.0, []
    R_avg = np.mean(R_values)
    spo2 = 104 - 17 * R_avg
    return max(0, min(100, spo2)), R_values

def calculate_heart_rate(ir_values, fs):
    ir_filtered = bandpass_filter(ir_values, 0.8, 2.5, fs)
    peaks, _ = find_peaks(ir_filtered, distance=int(0.5 * fs), prominence=0.4 * np.std(ir_filtered))
    duration_sec = len(ir_filtered) / fs
    heart_rate = (len(peaks) / duration_sec) * 60
    return round(heart_rate), ir_filtered, peaks

# === WiFi Data Fetch from Flask API ===
def read_ir_data():
    st.write(f"🔌 Fetching from: {API_URL}")
    ir_values = []
    red_values = []
    touched = False
    start_time = None

    MAX_SAMPLES = 600
    SAMPLE_INTERVAL = 1  # second

    while len(ir_values) < MAX_SAMPLES:
        try:
            res = requests.get(API_URL, timeout=5)
            if res.status_code != 200:
                st.warning("Server error. Retrying...")
                time.sleep(SAMPLE_INTERVAL)
                continue
            data = res.json().get("data", [])
            for sample in data:
                ir, red = sample
                if not touched and ir > TOUCH_THRESHOLD:
                    st.write("✋ Finger detected. Starting...")
                    touched = True
                    start_time = time.time()
                if touched:
                    ir_values.append(ir)
                    red_values.append(red)
            if touched and len(ir_values) >= MAX_SAMPLES:
                st.success(f"✅ Collected {len(ir_values)} IR samples.")
                break
        except Exception as e:
            st.error(f"Error fetching: {e}")
        time.sleep(SAMPLE_INTERVAL)

    return ir_values, red_values

def analyze_signal(ir_values, red_values, fs, name, age, gender):
    st.subheader("Respiratory Signal Analysis")
    filtered_ir = preprocess_signal(ir_values, fs)
    peaks = detect_breath_peaks(filtered_ir, fs)
    rr = math.ceil(len(peaks) * (60 / READ_DURATION))
    st.success(f"✅ Detected {len(peaks)} breaths in {READ_DURATION} sec → Estimated RR: {rr:.2f}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered_ir, label='Filtered IR Signal')
    ax.plot(peaks, np.array(filtered_ir)[peaks], 'ro', label='Detected Breaths')
    ax.axhline(np.mean(filtered_ir), color='gray', linestyle='--', label='Mean Line')
    ax.legend(); ax.grid(); st.pyplot(fig)

    st.subheader("Heart Rate Estimation")
    hr, hr_filtered, hr_peaks = calculate_heart_rate(ir_values, fs)
    st.success(f"❤️ Estimated Heart Rate: {hr} BPM")
    fig_hr, ax_hr = plt.subplots(figsize=(12, 5))
    ax_hr.plot(hr_filtered, label='Filtered IR (Heart)', alpha=0.8)
    ax_hr.plot(hr_peaks, hr_filtered[hr_peaks], 'rx', label='Detected Beats')
    ax_hr.legend(); ax_hr.grid(); st.pyplot(fig_hr)

    st.subheader("SpO₂ Estimation")
    spo2, R_list = calculate_spo2(np.array(ir_values), np.array(red_values), fs)
    if spo2 > 0:
        st.success(f"✅ Estimated SpO₂: {spo2:.2f}% using {len(R_list)} pulse cycles.")
    else:
        st.warning("⚠ Unable to estimate SpO₂ reliably.")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(ir_values, label='IR Raw', alpha=0.6)
    ax2.plot(red_values, label='RED Raw', alpha=0.6)
    ax2.legend(); ax2.grid(); st.pyplot(fig2)

    send_to_thingspeak(spo2, rr, hr, name, age, gender, "VX26ZPD2D2YK5JRJ")

# === MAIN APP ===
def main():
    st.title("🌡️ IoT Vital Sign Monitor (Cloud)")
    st.sidebar.header("👤 Patient Info")
    name = st.sidebar.text_input("Name")
    age = st.sidebar.text_input("Age")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

   if st.button("Start Data Collection"):
    st.session_state['ir_values'] = []
    st.session_state['red_values'] = []

   if st.button("🔁 Start New Patient Session"):
    requests.post("https://health-monitor-7lno.onrender.com/reset")
    st.success("🧹 Buffer cleared! Ready for new patient.")
    

    ir_values, red_values = read_ir_data()
    st.session_state['ir_values'] = ir_values
    st.session_state['red_values'] = red_values

    if not ir_values or not red_values:
        st.warning("⚠ No IR or RED data collected.")
        return

        fs = len(ir_values) / READ_DURATION
        pd.DataFrame(ir_values, columns=["IR"]).to_csv(SAVE_IR_PATH, index=False)
        pd.DataFrame(red_values, columns=["RED"]).to_csv(SAVE_RED_PATH, index=False)
        st.write(f"📁 Saved {len(ir_values)} IR samples to {SAVE_IR_PATH}")
        st.write(f"📁 Saved {len(red_values)} RED samples to {SAVE_RED_PATH}")
        analyze_signal(ir_values, red_values, fs, name, age, gender)

if __name__ == "__main__":
    main()
