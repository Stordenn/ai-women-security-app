import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from twilio.rest import Client
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation

current_timestamp = datetime.now()

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if 'model' not in st.session_state:
    st.session_state.model = load_model('bilstm_model_600.h5')

# Initialize session state variables
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'landmarks_buffer' not in st.session_state:
    st.session_state.landmarks_buffer = []
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Waiting"
if 'alert' not in st.session_state:
    st.session_state.alert = "None"
if 'lm_cnt' not in st.session_state:
    st.session_state.lm_cnt = []
if 'on_alert_message' not in st.session_state:
    st.session_state.on_alert_message=None
if 'label' not in st.session_state:
    st.session_state.label=None
if 'alert_msg_sent' not in st.session_state:
    st.session_state.alert_msg_sent=None    
if 'rescue_cnt' not in st.session_state:
    st.session_state.rescue_cnt=0
if 'current_datetime' not in st.session_state:
    st.session_state.current_datetime=None

# Function to predict the label based on the landmarks
def predict(padded_landmarks):
    # Load the trained BiLSTM model
    st.session_state.model = load_model('bilstm_model_600.h5')
    input_data = padded_landmarks.reshape(1, 500, 63)
    prediction = st.session_state.model.predict(input_data)
    print(prediction)
    #print(prediction[0][np.argmax(prediction[0])])
    st.session_state.model=False
    if prediction[0][0]> 0.99999:
        #predicted_label = np.argmax(prediction)
        st.session_state.label = 'Rescue'
        st.session_state.rescue_cnt+=1

        if st.session_state.on_alert_message:
            formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.current_datetime=formatted_timestamp

            sent_success=alert_message_on()
            st.session_state.alert_msg_sent=sent_success
        else:
            
            sent_fail=alert_message_off()
            st.session_state.alert_msg_sent=sent_fail

    elif prediction[0][1]> 0.999:
        st.session_state.label = 'Emergency'
    else:
        st.session_state.label = "Not Accurate"
    
    # Clear buffers after prediction
    st.session_state.landmarks_buffer.clear()
    st.session_state.lm_cnt.clear()


# Function to process video frames
def process_camera(placeholder,info_placeholder):
    
    while st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("Try again upload video or camera not accessible.")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.right_hand_landmarks is None:
            st.session_state.lm_cnt.append(0)
            st.session_state.current_status = "Waiting"

        if results.right_hand_landmarks:
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            if frame_landmarks is not None:
                st.session_state.landmarks_buffer.append(frame_landmarks)
                st.session_state.lm_cnt.append(1)
                st.session_state.current_status = "Collecting Landmarks"

        if len(st.session_state.lm_cnt) > 10 and sum(st.session_state.lm_cnt) > 10:
            if st.session_state.lm_cnt[-4:] == [1, 0, 0, 0]:
                if len(st.session_state.landmarks_buffer) < 500:
                    padded_landmarks = np.vstack(
                        [st.session_state.landmarks_buffer, np.zeros((500 - len(st.session_state.landmarks_buffer), 63))]
                    )
                else:
                    padded_landmarks = np.array(st.session_state.landmarks_buffer[-500:])
                if padded_landmarks.shape[0] == 500:
                    predict(padded_landmarks)
                    st.session_state.alert = st.session_state.label  # Update Alert with prediction result

        # Draw a box for status display
        frame_height, frame_width, _ = frame.shape
        box_height = 50
        cv2.rectangle(frame, (0, 0), (frame_width, box_height), (255, 255, 255), -1)
        cv2.rectangle(frame, (0, 0), (frame_width, box_height), (0, 0, 0), 2)

        # Display Current Status and Alert
        font_scale = 0.6
        font_thickness = 2
        cv2.putText(frame, "Current Status:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, st.session_state.current_status, (160, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

        cv2.putText(frame, "Alert:", (frame_width // 2 + 50, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, st.session_state.alert, (frame_width // 2 + 110, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

        # Display the frame
        placeholder.image(frame, channels="BGR", use_column_width=True)

        info_placeholder.info(st.session_state.alert_msg_sent)

        #if st.session_state.alert == 'Rescue':
            

# Streamlit UI
st.title("Silent Alert: Advancing Women's Security AI")




if 'placeholder' not in st.session_state:
    st.session_state.placeholder = st.empty()


default_image = np.zeros((400, 730, 3), dtype=np.uint8)

if 'info_placeholder' not in st.session_state:
    st.session_state.info_placeholder = st.empty()

if 'placeholder_initialized' not in st.session_state or not st.session_state.placeholder_initialized:
    st.session_state.placeholder.image(default_image, channels="RGB")
    st.session_state.placeholder_initialized = True

if 'info_placeholder_intialized' not in st.session_state or not st.session_state.info_placeholder_intialized:
    st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)
    st.session_state.info_placeholder_intialized=True

def start_camera():
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)

def stop_camera():
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        st.session_state.cap.release()
        st.session_state.cap = None
    # Reset placeholder to default image
    st.session_state.placeholder.image(default_image, channels="RGB")

    st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)

    st.session_state.current_status = "Waiting"
    st.session_state.alert = "None"
def alert_message_on():
    sent_cnt = st.session_state.rescue_cnt

    # Read Twilio secrets from .env file
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
    destination_phone_number = os.getenv("DESTINATION_PHONE_NUMBER")

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_=twilio_phone_number,
        body=f"""\n🚨 **Rescue Alert** 🚨\n
            \nDate & Time: {st.session_state.current_datetime}\n
            \nLocation: https://www.google.com/maps?q={st.session_state.latitude},{st.session_state.longitude}\n
            \nThank you.\n
            """,
        to=destination_phone_number
    )

    print(message.sid)
    return f'Alert successfully sent : {sent_cnt}'

def alert_message_off():
    #st.warning()
    return 'Message alert is off'

# Sidebar for buttons
st.sidebar.title("Alert Message")

if st.sidebar.button('On'):
    st.session_state.placeholder.image(default_image, channels="RGB")
    st.session_state.alert_msg_sent="Message alert is on"
    st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)
    st.session_state.on_alert_message=True
    

if st.sidebar.button('Off'):
    st.session_state.placeholder.image(default_image, channels="RGB")
    st.session_state.alert_msg_sent='Message alert is off'
    st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)
    st.session_state.on_alert_message=False

location = streamlit_geolocation()
if location:
    st.session_state.latitude = location["latitude"]
    st.session_state.longitude = location["longitude"]
else:
    st.session_state.latitude = None
    st.session_state.longitude = None

# Sidebar for buttons
st.sidebar.title("Webcam")

# Sidebar buttons
if st.sidebar.button("Start"):
    start_camera()
    process_camera(st.session_state.placeholder,st.session_state.info_placeholder)

if st.sidebar.button("Stop"):
    st.session_state.rescue_cnt=0
    st.session_state.alert_msg_sent=f'Alert successfully sent : {st.session_state.rescue_cnt}'
    st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)
    stop_camera()

# Function to process uploaded video
def process_uploaded_video(uploaded_file, placeholder, info_placeholder):
    st.session_state.cap = cv2.VideoCapture(uploaded_file)
    process_camera(placeholder, info_placeholder)
    st.session_state.cap.release()

# Sidebar for buttons
st.sidebar.title("Video Upload")   

# File uploader for video upload
uploaded_video = st.sidebar.file_uploader("Upload a video file", type=["mp4"])
if uploaded_video is not None:
    # Save uploaded video to a temporary file
    temp_file_path = f"temp_{uploaded_video.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.session_state.placeholder.image(default_image, channels="RGB")
    st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)
    # Process the uploaded video
    #st.sidebar.title("Process Uploaded Video")
    if st.sidebar.button("Start",key="123456"):
        
        process_uploaded_video(temp_file_path, st.session_state.placeholder, st.session_state.info_placeholder)
    if st.sidebar.button("Stop",key="56678543"):
        st.session_state.rescue_cnt=0
        st.session_state.alert_msg_sent=f'Alert successfully sent : {st.session_state.rescue_cnt}'
        st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)
        stop_camera()    

else:
    st.session_state.placeholder.image(default_image, channels="RGB")
    st.session_state.info_placeholder.info(st.session_state.alert_msg_sent)
#st.sidebar.text('🛡️ Save Humanity')
with st.sidebar.empty():
    st.markdown(
        """
        <div>
            <b>🛡️ Save Humanity 🛡️</b>
        </div>
        """,
        unsafe_allow_html=True,
    )    