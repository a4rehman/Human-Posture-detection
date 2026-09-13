import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def classify_posture(back_angle, vertical_diff):
    if vertical_diff < 0.08:
        return "Sleeping / Lying Down"
    if back_angle > 155:
        return "Standing"
    if 70 < back_angle < 130:
        return "Sitting"
    return "Analyzing..."


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


def process_frame_logic(frame, pose, mp_pose, mp_drawing):
    import cv2
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    posture_status = "Analyzing..."
    angles_info = {}

    try:
        landmarks = results.pose_landmarks.landmark

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

        back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        angles_info = {
            "Back Angle": round(back_angle, 1),
            "Neck Angle": round(neck_angle, 1),
            "Knee Angle": round(knee_angle, 1),
        }

        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_mid_y = (left_hip[1] + right_hip[1]) / 2
        vertical_diff = abs(shoulder_mid_y - hip_mid_y)

        posture_status = classify_posture(back_angle, vertical_diff)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(102, 126, 234), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(240, 147, 251), thickness=2, circle_radius=2)
        )

    except Exception:
        posture_status = "No Person Detected"

    return image, posture_status, angles_info