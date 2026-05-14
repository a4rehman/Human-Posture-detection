# AI Human Posture Detection 🧘

A professional computer vision application that detects and classifies human posture (Standing, Sitting, Sleeping) in real-time using Mediapipe and Streamlit.

## Features
- **Real-time Detection**: Uses webcam feed for live posture analysis.
- **Video Upload**: Supports analyzing pre-recorded video files (.mp4, .mov, .avi).
- **Landmark Visualization**: Displays 33 body landmarks for technical insight.
- **Smart Classification**: Heuristic-based classification for Standing, Sitting, and Lying down.

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Human-Posture-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## Legacy Support (Accelerometer Data)
The original project using PhonePi and accelerometer data is still available:
- Run `python PhonePi.py` to collect data.
- Run `python live_acc.py` for DTW-based classification.

## Tech Stack
- **Streamlit**: For the interactive web interface.
- **Mediapipe**: For high-fidelity body pose estimation.
- **OpenCV**: For video frame processing.
- **NumPy**: For mathematical calculations.

