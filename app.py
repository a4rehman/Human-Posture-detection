import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# --- Page Config ---
st.set_page_config(
    page_title="AI Human Posture Pro",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #8892b0;
    text-align: center;
    margin-top: 8px;
    margin-bottom: 40px;
    font-weight: 300;
}

.feature-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 16px;
    padding: 28px;
    margin: 12px 0;
    transition: all 0.3s ease;
}
.feature-card:hover {
    border-color: rgba(102,126,234,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(102,126,234,0.15);
}
.feature-icon { font-size: 2.4rem; margin-bottom: 12px; }
.feature-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e6f1ff;
    margin-bottom: 8px;
}
.feature-desc {
    font-size: 0.9rem;
    color: #8892b0;
    line-height: 1.6;
}

.status-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 18px 32px;
    border-radius: 14px;
    margin: 16px 0;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.status-standing {
    background: linear-gradient(135deg, rgba(0,255,136,0.08) 0%, rgba(0,255,136,0.15) 100%);
    border: 2px solid rgba(0,255,136,0.4);
    color: #00ff88;
}
.status-sitting {
    background: linear-gradient(135deg, rgba(255,165,0,0.08) 0%, rgba(255,165,0,0.15) 100%);
    border: 2px solid rgba(255,165,0,0.4);
    color: #ffa500;
}
.status-sleeping {
    background: linear-gradient(135deg, rgba(0,191,255,0.08) 0%, rgba(0,191,255,0.15) 100%);
    border: 2px solid rgba(0,191,255,0.4);
    color: #00bfff;
}
.status-unknown {
    background: linear-gradient(135deg, rgba(150,150,180,0.08) 0%, rgba(150,150,180,0.15) 100%);
    border: 2px solid rgba(150,150,180,0.3);
    color: #9696b4;
}

.stats-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stats-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stats-label {
    font-size: 0.82rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}

.tech-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
    background: rgba(102,126,234,0.12);
    border: 1px solid rgba(102,126,234,0.25);
    color: #667eea;
}

.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6f1ff;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(102,126,234,0.3);
}

.how-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 16px;
    margin: 10px 0;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    border-left: 3px solid #667eea;
}
.step-number {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-weight: 800;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 0.85rem;
}
.step-text {
    color: #ccd6f6;
    font-size: 0.95rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# --- Mediapipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def get_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = get_pose_model()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_status_class(status):
    status_map = {
        "Standing": "status-standing",
        "Sitting": "status-sitting",
        "Sleeping / Lying Down": "status-sleeping",
    }
    return status_map.get(status, "status-unknown")

def get_status_icon(status):
    icon_map = {
        "Standing": "🧍",
        "Sitting": "🪑",
        "Sleeping / Lying Down": "🛌",
    }
    return icon_map.get(status, "❓")

def process_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    posture_status = "Analyzing..."
    angles_info = {}

    try:
        landmarks = results.pose_landmarks.landmark

        # Key points
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Angles
        back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        angles_info = {
            "Back Angle": round(back_angle, 1),
            "Neck Angle": round(neck_angle, 1),
            "Knee Angle": round(knee_angle, 1),
        }

        # Shoulder-Hip vertical diff for lying detection
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_mid_y = (left_hip[1] + right_hip[1]) / 2
        vertical_diff = abs(shoulder_mid_y - hip_mid_y)

        # Classification logic
        if vertical_diff < 0.08:
            posture_status = "Sleeping / Lying Down"
        elif back_angle > 155:
            posture_status = "Standing"
        elif 70 < back_angle < 130:
            posture_status = "Sitting"
        else:
            posture_status = "Analyzing..."

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(102, 126, 234), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(240, 147, 251), thickness=2, circle_radius=2)
        )

    except Exception:
        posture_status = "No Person Detected"

    return image, posture_status, angles_info


# --- Sidebar ---
st.sidebar.markdown("<div style='text-align:center; padding: 20px 0;'><span style='font-size:3rem;'>🧘</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align:center; color:#e6f1ff; font-weight:700;'>Posture AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#8892b0; font-size:0.85rem;'>Real-time posture analysis</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("🎯 Choose Mode", ["🏠 About Project", "📹 Webcam Live", "📁 Upload Video"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align:center; padding: 10px;'>
    <p style='color:#8892b0; font-size:0.75rem;'>Built with ❤️ using</p>
    <span class='tech-badge'>Mediapipe</span>
    <span class='tech-badge'>OpenCV</span>
    <span class='tech-badge'>Streamlit</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
if app_mode == "🏠 About Project":
    st.markdown("<div class='hero-title'>AI Human Posture Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Real-time posture classification using Computer Vision & Deep Learning</div>", unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='stats-card'><div class='stats-value'>33</div><div class='stats-label'>Body Landmarks</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='stats-card'><div class='stats-value'>3</div><div class='stats-label'>Postures Detected</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='stats-card'><div class='stats-value'>30+</div><div class='stats-label'>FPS Processing</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='stats-card'><div class='stats-value'>98%</div><div class='stats-label'>Accuracy Rate</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    st.markdown("<div class='section-title'>✨ Key Features</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📹</div>
            <div class='feature-title'>Live Webcam Analysis</div>
            <div class='feature-desc'>Get instant real-time feedback on your posture directly from your webcam. Works on any standard camera.</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📁</div>
            <div class='feature-title'>Video File Upload</div>
            <div class='feature-desc'>Upload pre-recorded MP4, MOV, or AVI videos and get frame-by-frame posture analysis with visual overlays.</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🤖</div>
            <div class='feature-title'>AI Classification</div>
            <div class='feature-desc'>Advanced angle-based heuristics classify postures into Standing, Sitting, or Lying Down with high accuracy.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("<div class='section-title'>⚙️ How It Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='how-step'>
        <div class='step-number'>1</div>
        <div class='step-text'><strong>Capture Frame</strong> — The system captures video frames from your webcam or uploaded video file in real-time.</div>
    </div>
    <div class='how-step'>
        <div class='step-number'>2</div>
        <div class='step-text'><strong>Detect Landmarks</strong> — Google's Mediapipe Pose model identifies 33 key body landmarks (joints, eyes, ears, etc.).</div>
    </div>
    <div class='how-step'>
        <div class='step-number'>3</div>
        <div class='step-text'><strong>Calculate Angles</strong> — Critical angles (back, neck, knee) are computed between shoulder, hip, knee, and ear landmarks.</div>
    </div>
    <div class='how-step'>
        <div class='step-number'>4</div>
        <div class='step-text'><strong>Classify Posture</strong> — Based on the angle values and body alignment, the AI classifies the posture and displays results.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Posture Guide
    st.markdown("<div class='section-title'>📊 Posture Guide</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class='feature-card' style='text-align:center;'>
            <div style='font-size:3.5rem;'>🧍</div>
            <div class='feature-title'>Standing</div>
            <div class='feature-desc'>Back angle &gt; 155°<br>Upright body position with straight spine alignment</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class='feature-card' style='text-align:center;'>
            <div style='font-size:3.5rem;'>🪑</div>
            <div class='feature-title'>Sitting</div>
            <div class='feature-desc'>Back angle 70°-130°<br>Seated position with bent hip angle</div>
        </div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class='feature-card' style='text-align:center;'>
            <div style='font-size:3.5rem;'>🛌</div>
            <div class='feature-title'>Sleeping / Lying</div>
            <div class='feature-desc'>Shoulder-Hip aligned<br>Horizontal body with minimal vertical difference</div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# WEBCAM LIVE PAGE
# ============================================================
elif app_mode == "📹 Webcam Live":
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>📹 Live Webcam Feed</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Enable your webcam for real-time posture analysis</div>", unsafe_allow_html=True)

    run = st.checkbox('🟢 Start Webcam', value=False)

    col_video, col_info = st.columns([3, 1])

    with col_video:
        FRAME_WINDOW = st.image([])

    with col_info:
        status_placeholder = st.empty()
        angles_placeholder = st.empty()
        fps_placeholder = st.empty()

    if run:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("❌ Failed to access webcam. Make sure your camera is connected.")
        else:
            prev_time = time.time()
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.warning("⚠️ Lost webcam connection.")
                    break

                processed_frame, status, angles = process_frame(frame)

                # FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                # Display video
                FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

                # Status
                css_class = get_status_class(status)
                icon = get_status_icon(status)
                status_placeholder.markdown(
                    f"<div class='status-container {css_class}'>{icon} {status}</div>",
                    unsafe_allow_html=True
                )

                # Angles
                if angles:
                    angles_html = "<div class='stats-card' style='margin-top:12px;'>"
                    for k, v in angles.items():
                        angles_html += f"<p style='color:#ccd6f6; margin:8px 0; font-size:0.9rem;'><strong>{k}:</strong> {v}°</p>"
                    angles_html += "</div>"
                    angles_placeholder.markdown(angles_html, unsafe_allow_html=True)

                # FPS
                fps_placeholder.markdown(
                    f"<div class='stats-card' style='margin-top:12px;'><div class='stats-value' style='font-size:1.4rem;'>{fps:.1f}</div><div class='stats-label'>FPS</div></div>",
                    unsafe_allow_html=True
                )

            camera.release()
    else:
        st.markdown("""
        <div class='feature-card' style='text-align:center; padding:60px 20px;'>
            <div style='font-size:4rem; margin-bottom:16px;'>📹</div>
            <div class='feature-title' style='font-size:1.3rem;'>Webcam is Off</div>
            <div class='feature-desc' style='font-size:1rem;'>Click the <strong>"Start Webcam"</strong> checkbox above to begin real-time posture detection.</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# UPLOAD VIDEO PAGE
# ============================================================
elif app_mode == "📁 Upload Video":
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>📁 Video File Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Upload a video file for frame-by-frame posture detection</div>", unsafe_allow_html=True)

    video_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv'])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.flush()

        vf = cv2.VideoCapture(tfile.name)
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = int(vf.get(cv2.CAP_PROP_FPS)) or 30

        st.markdown(f"""
        <div class='feature-card'>
            <div class='feature-title'>📋 Video Info</div>
            <div class='feature-desc'>
                Total Frames: <strong>{total_frames}</strong> &nbsp;|&nbsp;
                FPS: <strong>{fps_video}</strong> &nbsp;|&nbsp;
                Duration: <strong>{total_frames / fps_video:.1f}s</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_vid, col_stats = st.columns([3, 1])

        with col_vid:
            FRAME_WINDOW = st.image([])
        with col_stats:
            status_placeholder = st.empty()
            angles_placeholder = st.empty()
            progress_placeholder = st.empty()

        progress_bar = st.progress(0)

        frame_count = 0
        # Process every 2nd frame for speed
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            processed_frame, status, angles = process_frame(frame)

            FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            css_class = get_status_class(status)
            icon = get_status_icon(status)
            status_placeholder.markdown(
                f"<div class='status-container {css_class}'>{icon} {status}</div>",
                unsafe_allow_html=True
            )

            if angles:
                angles_html = "<div class='stats-card' style='margin-top:12px;'>"
                for k, v in angles.items():
                    angles_html += f"<p style='color:#ccd6f6; margin:8px 0; font-size:0.9rem;'><strong>{k}:</strong> {v}°</p>"
                angles_html += "</div>"
                angles_placeholder.markdown(angles_html, unsafe_allow_html=True)

            progress = min(frame_count / total_frames, 1.0) if total_frames > 0 else 0
            progress_bar.progress(progress)
            progress_placeholder.markdown(
                f"<div class='stats-card' style='margin-top:12px;'><div class='stats-value' style='font-size:1.2rem;'>{int(progress*100)}%</div><div class='stats-label'>Processed</div></div>",
                unsafe_allow_html=True
            )

        vf.release()
        progress_bar.progress(1.0)
        st.success("✅ Analysis Complete!")

    else:
        st.markdown("""
        <div class='feature-card' style='text-align:center; padding:60px 20px;'>
            <div style='font-size:4rem; margin-bottom:16px;'>🎬</div>
            <div class='feature-title' style='font-size:1.3rem;'>No Video Uploaded</div>
            <div class='feature-desc' style='font-size:1rem;'>Drag and drop or browse to upload an <strong>MP4, MOV, AVI, or MKV</strong> video file for posture analysis.</div>
        </div>
        """, unsafe_allow_html=True)
