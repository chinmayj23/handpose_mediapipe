import cv2
import mediapipe as mp
import numpy as np
import math
import json
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

def capture_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap



def detect_hand_keypoints(frame):  
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    return results

def calculate_angle(A, B, C):
    AB = np.array([A[0] - B[0], A[1] - B[1], A[2] - B[2]])
    BC = np.array([C[0] - B[0], C[1] - B[1], C[2] - B[2]])
    dot_product = np.dot(AB, BC)
    mod_AB = np.linalg.norm(AB)
    mod_BC = np.linalg.norm(BC)
    cos_theta = dot_product / (mod_AB * mod_BC)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

def draw_hand_keypoints(frame, hand_landmarks):
    
    mp_drawing.draw_landmarks(
        frame, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5)
    )
    
    # Labeling each joint according to the doc
    for idx, landmark in enumerate(hand_landmarks.landmark):
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def save_joint_angles(joint_angles, filename="joint_angles.json"):
    with open(filename, 'w') as f:
        json.dump(joint_angles, f, indent=4)

def save_output_video(frames, output_path="output_video.mp4", fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def main(video_path):
    cap = capture_video(video_path)
    if not cap:
        return

    joint_angles = []
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_hand_keypoints(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame = draw_hand_keypoints(frame, hand_landmarks)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                
                # Calculating angles for each set of three joints
                angles = {}
                for i in range(1, len(landmarks) - 1):
                    A, B, C = landmarks[i - 1], landmarks[i], landmarks[i + 1]
                    angle = calculate_angle(A, B, C)
                    angles[f"angle_{i}"] = angle
                
                joint_angles.append(angles)
        
        frames.append(frame)
    
    cap.release()
    save_joint_angles(joint_angles)
    save_output_video(frames)

if __name__ == "__main__":
    video_path = "PXL_20240722_095208762.TS.mp4"
    main(video_path)

