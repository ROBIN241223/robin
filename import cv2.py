import cv2
import face_recognition
import numpy as np
import os
import mediapipe as mp
import datetime

# MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_time_of_day():
    """Xác định thời gian (sáng, trưa, chiều, tối)."""
    hour = datetime.datetime.now().hour
    if 6 <= hour < 12:  return "morning"
    if 12 <= hour < 18: return "afternoon"
    if 18 <= hour < 21: return "evening"
    return "night"

def estimate_lighting(image):
    """Ước lượng ánh sáng (rất tối, tối, trung bình, sáng, rất sáng)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < 30:    return "very_dark"
    if avg_brightness < 60:    return "dark"
    if avg_brightness < 180:   return "medium"
    if avg_brightness < 220:   return "bright"
    return "very_bright"

def enhance_image(image, time_of_day=None, lighting=None):
    """Tăng cường ảnh dựa trên thời gian và ánh sáng (dùng cho pose và đặc trưng)."""
    enhanced = image.copy()
    clip_limit = 4.0 if lighting == "very_dark" else 2.0 if lighting == "very_bright" else 3.0

    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)

    # Cân bằng trắng (white balance)
    if lighting in ["dark", "very_dark"]:
        avg_rgb = np.mean(enhanced, axis=(0, 1))
        enhanced = enhanced * (128.0 / avg_rgb)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    enhanced = cv2.addWeighted(enhanced, 1.5, cv2.GaussianBlur(enhanced, (0, 0), 3), -0.5, 0) # Sharpen
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)  # Denoise

    if time_of_day == "night":
        enhanced = cv2.add(enhanced, (30, 30, 30, 0))
    elif time_of_day == "evening":
        h, s, v = cv2.split(cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV))
        enhanced = cv2.cvtColor(cv2.merge((cv2.add(h, -10), cv2.add(s, 20), v)), cv2.COLOR_HSV2RGB)

    if lighting in ["dark", "very_dark"]:
      enhanced = cv2.add(enhanced, (70 if lighting == "very_dark" else 50,)*3 + (0,))

    return enhanced


def enhance_image_for_face_recognition(image, lighting=None):
    """Tăng cường ảnh cho nhận diện khuôn mặt."""
    enhanced = image.copy()
    clip_limit = 4.0 if lighting == "very_dark" else 2.0 if lighting == "very_bright" else 3.0

    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)

    # Cân bằng trắng
    if lighting in ["dark", "very_dark"]:
        avg_rgb = np.mean(enhanced, axis=(0, 1))
        enhanced = enhanced * (128.0 / avg_rgb)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    enhanced = cv2.addWeighted(enhanced, 1.5, cv2.GaussianBlur(enhanced, (0, 0), 3), -0.5, 0)

    if lighting in ["dark", "very_dark", "very_bright"]:
        gamma = 1.2 if lighting in ["dark", "very_dark"] else 0.8
        enhanced = (cv2.pow(enhanced.astype(np.float32) / 255.0, gamma) * 255).astype(np.uint8)

    return enhanced

def extract_hair_features(image, face_location):
    """Trích xuất đặc trưng tóc (màu, kiểu)."""
    top, right, bottom, left = face_location
    hair_top = max(0, top - int((bottom - top) * 0.3))
    hair_region = image[hair_top:top, left:right]
    if hair_region.size > 0:
        return {'color': np.mean(hair_region, axis=(0, 1)), 'style': 'long' if (top - hair_top) > (bottom - top) * 0.4 else 'short'}
    return {'color': None, 'style': None}

def extract_glasses_features(image, face_location):
    """Trích xuất đặc trưng kính (có/không)."""
    top, right, bottom, left = face_location
    eyes_region = image[top:top + (bottom - top) // 2, left:right]
    if eyes_region.size > 0:
        edges = cv2.Canny(cv2.cvtColor(eyes_region, cv2.COLOR_RGB2GRAY), 50, 150)
        return {'present': np.sum(edges > 0) / float(edges.size) > 0.05}
    return {'present': False}

def calculate_hair_similarity(hair1, hair2):
    """So sánh đặc trưng tóc."""
    if not all([hair1, hair2, hair1['color'], hair2['color']]): return 0.0
    color_similarity = 1.0 / (1.0 + np.linalg.norm(hair1['color'] - hair2['color']))
    style_similarity = 1.0 if hair1['style'] == hair2['style'] else 0.5
    return 0.7 * color_similarity + 0.3 * style_similarity

def calculate_glasses_similarity(glasses1, glasses2):
    """So sánh đặc trưng kính."""
    return 1.0 if glasses1 and glasses2 and glasses1['present'] == glasses2['present'] else 0.0

def calculate_pose_similarity(pose1_landmarks, pose2_landmarks):
    """Tính độ tương đồng dáng người."""
    if pose1_landmarks is None or pose2_landmarks is None: return 0.0
    important_joints = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                      mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
    distances, weights = [], []

    for joint in important_joints:
        x1, y1, _, v1 = pose1_landmarks[joint]
        x2, y2, _, v2 = pose2_landmarks[joint]
        distances.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        weights.append((v1 + v2) / 2.0 * 0.7 + 0.3)

    if not distances: return 0.0
    weighted_distances = [d * w for d, w in zip(distances, weights)]
    avg_distance = sum(weighted_distances) / sum(weights) if sum(weights) > 0 else 1.0
    return 1.0 / (1.0 + avg_distance)


def load_known_data(directory):
    """Tải dữ liệu, trích xuất và ước lượng môi trường."""
    known_face_encodings, known_poses, known_hair, known_glasses, known_environments = [], [], [], [], []
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory, filename)
            try:
                image = cv2.cvtColor(face_recognition.load_image_file(image_path), cv2.COLOR_BGR2RGB)
                time_of_day, lighting = get_time_of_day(), estimate_lighting(image)
                known_environments.append({'time': time_of_day, 'lighting': lighting})

                image_enhanced = enhance_image(image, time_of_day, lighting)
                image_enhanced_face = enhance_image_for_face_recognition(image, lighting)

                face_locations = face_recognition.face_locations(image_enhanced_face)
                face_encodings = face_recognition.face_encodings(image_enhanced_face, face_locations)

                if face_encodings:
                    for i, face_encoding in enumerate(face_encodings):
                        known_face_encodings.append(face_encoding)
                        known_hair.append(extract_hair_features(image_enhanced, face_locations[i]))
                        known_glasses.append(extract_glasses_features(image_enhanced, face_locations[i]))
                else:
                    print(f"Cảnh báo: Không tìm thấy khuôn mặt trong {filename}")

                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3) as pose:
                    results = pose.process(image_enhanced)
                    if results.pose_landmarks:
                        landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                        known_poses.append(landmarks)
                    else:
                        print(f"Cảnh báo: Không tìm thấy dáng trong {filename}")
                        known_poses.append(None)
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")
    return known_face_encodings, known_poses, known_hair, known_glasses, known_environments

def find_wife_in_harsh_lighting(known_face_encodings, known_poses, known_hair, known_glasses, known_environments):
    """Tìm vợ, xét môi trường và ánh sáng."""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Không thể mở webcam.")
        return

    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Lỗi đọc frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time, current_lighting = get_time_of_day(), estimate_lighting(rgb_frame)

            # Tăng cường ảnh
            enhanced_frame_face = enhance_image_for_face_recognition(rgb_frame, current_lighting)
            enhanced_frame_pose_features = enhance_image(rgb_frame, current_time, current_lighting)

            # Dáng
            results = pose.process(enhanced_frame_pose_features)
            current_pose_landmarks = None
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                current_pose_landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]

            # Khuôn mặt
            face_locations = face_recognition.face_locations(enhanced_frame_face)
            face_encodings = face_recognition.face_encodings(enhanced_frame_face, face_locations)

            best_pose_sim, best_hair_sim, best_glasses_sim, best_face_dist, best_env_sim = 0.0, 0.0, 0.0, float('inf'), 0.0

            if current_pose_landmarks:
                for known_pose in known_poses:
                    if known_pose:
                        pose_sim = calculate_pose_similarity(known_pose, current_pose_landmarks)
                        best_pose_sim = max(best_pose_sim, pose_sim)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                current_hair = extract_hair_features(enhanced_frame_pose_features, (top, right, bottom, left))
                current_glasses = extract_glasses_features(enhanced_frame_pose_features, (top, right, bottom, left))

                if best_pose_sim < 0.5:
                    for known_h, known_g, env in zip(known_hair, known_glasses, known_environments):
                        best_hair_sim = max(best_hair_sim, calculate_hair_similarity(known_h, current_hair))
                        best_glasses_sim = max(best_glasses_sim, calculate_glasses_similarity(known_g, current_glasses))
                        env_sim = (1.0 if env['time'] == current_time else 0.0) + (1.0 if env['lighting'] == current_lighting else 0.0)
                        best_env_sim = max(best_env_sim, env_sim)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        if face_distances.size > 0:
                            best_face_dist = min(best_face_dist, min(face_distances))

                        name, color, text = "Unknown", (0, 0, 255), "Unknown"
                        if best_pose_sim > 0.5:
                            name, color = "Vo", (0, 255, 0)
                            text = f"Vo (Dang: {best_pose_sim:.2f}"
                            if best_env_sim > 0.7 or best_hair_sim > 0.6 or best_glasses_sim > 0.6 or best_face_dist < 0.7:
                                text += ", "
                            if best_env_sim > 0.7: text += f"MT: {best_env_sim:.2f}, "
                            if best_hair_sim > 0.6:  text += f"Toc: {best_hair_sim:.2f}, "
                            if best_glasses_sim > 0.6: text += f"Kinh: {best_glasses_sim:.2f}, "
                            if best_face_dist < 0.7: text += f"Mat: {best_face_dist:.2f}"
                            text += ")"

                        elif best_env_sim > 0.7:
                            name, color = "Vo", (0, 255, 255)
                            text = f"Vo (MT: {best_env_sim:.2f}"
                            if best_hair_sim > 0.6 or best_glasses_sim > 0.6 or best_face_dist < 0.7:
                                text += ", "
                            if best_hair_sim > 0.6: text += f"Toc: {best_hair_sim:.2f}, "
                            if best_glasses_sim > 0.6: text += f"Kinh: {best_glasses_sim:.2f}, "
                            if best_face_dist < 0.7: text += f"Mat: {best_face_dist:.2f}"
                            text += ")"

                        elif best_hair_sim > 0.6 or best_glasses_sim > 0.6 or best_face_dist < 0.7:
                            name, color = "Co The La Vo", (255, 255, 0)
                            text = f"Co The La Vo ("
                            if best_hair_sim > 0.6:
                                text += f"Toc: {best_hair_sim:.2f}, "
                            if best_glasses_sim > 0.6:
                                text += f"Kinh: {best_glasses_sim:.2f}, "
                            if best_face_dist < 0.7:
                                text += f"Mat: {best_face_dist:.2f}"
                            text += ")"

                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                    cv2.LINE_AA)

                    cv2.imshow('Webcam - Tim Vo', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()

            if __name__ == "__main__":
                known_faces_dir = "known_faces"
                known_face_encodings, known_poses, known_hair_features, known_glasses_features, known_environments = load_known_data(
                    known_faces_dir)
                find_wife_in_harsh_lighting(known_face_encodings, known_poses, known_hair_features,
                                            known_glasses_features,
                                            known_environments)