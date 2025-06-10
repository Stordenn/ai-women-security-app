# 🛡️ Silent Alert — Advancing Women's Security AI

An AI-powered real-time system to detect distress gestures and silently send rescue alerts with geolocation, helping enhance women's personal safety.

---

## 🚀 Project Overview

This project is an **AI-based Women's Security Application** that uses computer vision and deep learning to:
✅ Detect predefined distress gestures from video streams  
✅ Classify gestures using a trained **BiLSTM neural network**  
✅ Trigger **silent SMS alerts** with GPS location to predefined emergency contacts  

Built using **MediaPipe**, **TensorFlow/Keras**, **OpenCV**, **Twilio API**, and **Streamlit**.

---

## 🎯 Key Features

- Real-time video processing to detect gestures
- Trained BiLSTM model for distress gesture classification (~95% accuracy)
- Silent SMS alerts with GPS coordinates sent via Twilio
- Streamlit-based user interface
- Works with both **webcam** and **uploaded videos**
- Toggle ON/OFF alert mode via UI
- Geolocation fetched via `streamlit-geolocation` package

---

## 🛠️ Tech Stack

- **MediaPipe Holistic** → Body & Hand landmark detection
- **BiLSTM (TensorFlow/Keras)** → Gesture classification
- **OpenCV** → Video stream processing
- **Streamlit** → Frontend UI
- **Twilio API** → SMS alerts with location
- **Python** → Core logic
- **Streamlit Geolocation** → To fetch user's latitude and longitude

---

## 🔐 Environment Variables

Create a `.env` file (see `.env.example`) with:

```txt
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_number_here
DESTINATION_PHONE_NUMBER=destination_number_here
