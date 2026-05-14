import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="AI Human Posture Pro",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        color: #ffffff;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    .good-posture {
        background-color: #00ff0022;
        border: 2px solid #00ff00;
        color: #00ff00;
    }
    .bad-posture {
        background-color: #ff000022;
        border: 2px solid #ff0000;
        color: #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Mediapipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_frame(frame):
    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    results = pose.process(image)
    
    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    posture_status = "Unknown"
    color = (255, 255, 255)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        
        # Calculate angles
        # 1. Back Angle (Shoulder-Hip-Knee)
        back_angle = calculate_angle(shoulder, hip, knee)
        
        # 2. Neck Angle (Ear-Shoulder-Hip)
        neck_angle = calculate_angle(ear, shoulder, hip)
        
        # Heuristics for classification
        if back_angle > 150:
            posture_status = "Standing"
            color = (0, 255, 0)
        elif 70 < back_angle < 110:
            posture_status = "Sitting"
            color = (255, 165, 0)
        elif back_angle < 45 or back_angle > 160: # Rough approximation for lying down
             # Check alignment of shoulder and hip y-coordinates
             if abs(shoulder[1] - hip[1]) < 0.1:
                 posture_status = "Sleeping/Lying"
                 color = (0, 191, 255)
             else:
                 posture_status = "Moving"
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
    except Exception as e:
        pass
        
    return image, posture_status

# --- Sidebar ---
st.sidebar.title("Settings")
app_mode = st.sidebar.selectbox("Choose Mode", ["About Project", "Webcam Live", "Upload Video"])

if app_mode == "About Project":
    st.title("Human Posture Detection 🧘")
    st.markdown("""
    Welcome to the **Human Posture Detection** app. This system uses Computer Vision and Mediapipe to analyze body alignment in real-time.
    
    ### Features:
    - **Real-time Webcam Analysis**: Get instant feedback on your posture.
    - **Video Upload**: Analyze pre-recorded videos.
    - **Posture Classification**: Detects if you are Standing, Sitting, or Lying down.
    
    ### How it works:
    The app uses the **Mediapipe Pose** model to identify 33 key body landmarks. It then calculates the angles between specific points (like shoulders, hips, and knees) to determine your posture.
    """)
    st.image("https://mediapipe.dev/images/mobile/pose_classification_android_gui.gif", caption="Mediapipe Pose Landmark Detection")

elif app_mode == "Webcam Live":
    st.title("Live Webcam Feed")
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        if frame is None:
            st.error("Failed to access webcam.")
            break
            
        processed_frame, status = process_frame(frame)
        
        # Display results
        FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        
        status_text.markdown(f"<div class='status-box' style='color: white; border: 2px solid white;'>Detected Posture: {status}</div>", unsafe_allow_html=True)
    else:
        st.write('Stopped')
        camera.release()

elif app_mode == "Upload Video":
    st.title("Video File Analysis")
    video_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
    
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        st.write("Processing Video...")
        
        FRAME_WINDOW = st.image([])
        status_text = st.empty()
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            processed_frame, status = process_frame(frame)
            FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            status_text.markdown(f"<div class='status-box' style='color: white; border: 2px solid white;'>Detected Posture: {status}</div>", unsafe_allow_html=True)
            
        vf.release()
        st.success("Analysis Complete!")
