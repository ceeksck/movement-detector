import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

# Streamlit UI
st.title("🕺 Real-Time Movement Detector with Calories Estimation 🔥")

# Start webcam button
start_button = st.button("Start Webcam")

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# User weight input
user_weight = st.number_input("Enter Your Weight (kg)", min_value=30, max_value=150, value=70)

# Movement Counters
movement_counts = {
    "Left Arm Raise": 0,
    "Right Arm Raise": 0,
    "Squat": 0,
    "Left Knee Raise": 0,
    "Right Knee Raise": 0
}

# Movement State
movement_state = {
    "left_arm": None,
    "right_arm": None,
    "squat": None,
    "left_knee": None,
    "right_knee": None
}

# MET Values for Calories Estimation
MET_VALUES = {
    "Left Arm Raise": 3.0,
    "Right Arm Raise": 3.0,
    "Squat": 5.5,
    "Left Knee Raise": 4.0,
    "Right Knee Raise": 4.0
}

# Open webcam if "Start Webcam" is clicked
if start_button:
    st.write("📷 **Webcam Active** (Press 'q' to stop)")
    cap = cv2.VideoCapture(0)

    # Start time for session
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Draw Skeleton
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract Key Landmarks
            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
            right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y

            # Movement Detections
            if left_wrist_y < left_shoulder_y:
                if movement_state["left_arm"] != "up":
                    movement_state["left_arm"] = "up"
            else:
                if movement_state["left_arm"] == "up":
                    movement_state["left_arm"] = "down"
                    movement_counts["Left Arm Raise"] += 1

            if right_wrist_y < right_shoulder_y:
                if movement_state["right_arm"] != "up":
                    movement_state["right_arm"] = "up"
            else:
                if movement_state["right_arm"] == "up":
                    movement_state["right_arm"] = "down"
                    movement_counts["Right Arm Raise"] += 1

            if hip_y > hip_y + 0.15:
                if movement_state["squat"] != "down":
                    movement_state["squat"] = "down"
            else:
                if movement_state["squat"] == "down":
                    movement_state["squat"] = "up"
                    movement_counts["Squat"] += 1

            if left_knee_y < hip_y - 0.10:
                if movement_state["left_knee"] != "up":
                    movement_state["left_knee"] = "up"
            else:
                if movement_state["left_knee"] == "up":
                    movement_state["left_knee"] = "down"
                    movement_counts["Left Knee Raise"] += 1

            if right_knee_y < hip_y - 0.10:
                if movement_state["right_knee"] != "up":
                    movement_state["right_knee"] = "up"
            else:
                if movement_state["right_knee"] == "up":
                    movement_state["right_knee"] = "down"
                    movement_counts["Right Knee Raise"] += 1

        # Convert frame for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary Report
    elapsed_time = (time.time() - start_time) / 60
    total_calories = sum(MET_VALUES[move] * user_weight * elapsed_time * 3.5 * 5 / 1000 for move in movement_counts)

    st.subheader("📊 **Exercise Summary**")
    st.write(f"🕒 **Total Time:** {elapsed_time:.2f} minutes")
    st.write(f"🔥 **Total Calories Burned:** {total_calories:.2f} kcal")
    for move, count in movement_counts.items():
        st.write(f"**{move}:** {count} reps")
