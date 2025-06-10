# ai-women-security-app
One of my most impactful AI/ML projects is an AI-powered Women’s Security Application aimed at enhancing personal safety. The goal was to develop a system capable of detecting predefined distress gestures in real time and silently triggering emergency alerts with GPS location.

I built this project using a combination of computer vision and deep learning techniques. For real-time pose and hand gesture detection, I used MediaPipe Holistic, which provides accurate body and hand landmarks. These landmarks were processed through a Bidirectional LSTM (BiLSTM) neural network trained to classify distress gestures from live video frames. I developed the frontend using Streamlit to provide a simple and accessible interface, and integrated the Twilio API to send silent SMS alerts containing geolocation data to predefined emergency contacts.

My contributions included designing the full system architecture, implementing the video processing pipeline, training and optimizing the BiLSTM model, building the user interface, and integrating the alert system. The prototype achieves ~95% accuracy in detecting gestures and triggers alerts in under 2 seconds.

This project taught me how to transform an AI model into a real-world, user-focused product and strengthened my ability to build end-to-end AI solutions with meaningful social impact.
