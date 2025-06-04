import cv2
import face_recognition
import numpy as np
import os
import mediapipe as mp
import datetime  # Thêm thư viện datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def get_time_of_day():
    """Xác định thời gian trong ngày (sáng, trưa, chiều, tối)."""
    now = datetime.datetime.now()
    hour = now.hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 21:
        return "evening"
    else:
        return "night"


def estimate_lighting(image):
    """Ước lượng điều kiện ánh sáng (tối, sáng, trung bình)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    average_brightness = np.mean(gray)

    if average_brightness < 60:
        return "dark"
    elif average_brightness < 180:
        return "medium"
    else:
        return "bright"


def enhance_image(image, time_of_day=None, lighting=None):
    """
    Áp dụng các bộ lọc, điều chỉnh dựa trên thời gian và ánh sáng.
    """
    # 1. Tăng cường độ tương phản (CLAHE) - Luôn áp dụng
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Có thể điều chỉnh clipLimit
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # 2. Làm nét (Unsharp Masking) - Luôn áp dụng
    blurred = cv2.GaussianBlur(enhanced_image, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced_image, 1.5, blurred, -0.5, 0)

    # 3. Khử nhiễu (Non-local Means) - Luôn áp dụng
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)  # Có thể điều chỉnh thông số

    # 4. Điều chỉnh màu sắc và độ sáng (dựa trên thời gian và ánh sáng)
    if time_of_day == "night":
        # Tăng độ sáng cho ảnh chụp ban đêm
        denoised = cv2.add(denoised, (30, 30, 30, 0))  # Điều chỉnh độ sáng cộng thêm
    elif time_of_day == "evening":
        # Tăng cường màu ấm cho ảnh chụp buổi tối.
        hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.add(h, -10)  # Dịch chuyển hue về phía màu vàng/cam
        s = cv2.add(s, 20)
        enhanced_hsv = cv2.merge((h, s, v))
        denoised = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

    if lighting == "dark":
        denoised = cv2.add(denoised, (50, 50, 50, 0))
    # Không cần điều chỉnh đặc biệt cho "medium" và "bright"

    return denoised


def extract_hair_features(image, face_location):
    """Trích xuất đặc trưng tóc (màu sắc, kiểu tóc)."""
    top, right, bottom, left = face_location
    hair_features = {}
    hair_top = max(0, top - int((bottom - top) * 0.3))
    hair_region = image[hair_top:top, left:right]

    if hair_region.size > 0:
        avg_color = np.mean(hair_region, axis=(0, 1))
        hair_features['color'] = avg_color
        hair_length = top - hair_top
        face_height = bottom - top
        if hair_length > face_height * 0.4:
            hair_features['style'] = 'long'
        else:
            hair_features['style'] = 'short'
    else:
        hair_features['color'] = None
        hair_features['style'] = None
    return hair_features


def extract_glasses_features(image, face_location):
    """Trích xuất đặc trưng kính (có/không)."""
    top, right, bottom, left = face_location
    glasses_features = {}
    eye_top = top
    eye_bottom = top + (bottom - top) // 2
    eye_left = left
    eye_right = right

    eyes_region = image[eye_top:eye_bottom, eye_left:eye_right]

    if eyes_region.size > 0:
        gray_eyes = cv2.cvtColor(eyes_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_eyes, 50, 150)
        edge_density = np.sum(edges > 0) / float(edges.size)
        if edge_density > 0.05:
            glasses_features['present'] = True
        else:
            glasses_features['present'] = False
    else:
        glasses_features['present'] = False

    return glasses_features


def calculate_hair_similarity(hair1, hair2):
    """So sánh đặc trưng tóc."""
    if hair1 is None or hair2 is None or hair1['color'] is None or hair2['color'] is None:
        return 0.0

    color_similarity = 0.0
    style_similarity = 0.0
    if hair1['color'] is not None and hair2['color'] is not None:
        color_distance = np.linalg.norm(hair1['color'] - hair2['color'])
        color_similarity = 1.0 / (1.0 + color_distance)

    if hair1['style'] is not None and hair2['style'] is not None:
        if hair1['style'] == hair2['style']:
            style_similarity = 1.0
        else:
            style_similarity = 0.5
    return 0.7 * color_similarity + 0.3 * style_similarity


def calculate_glasses_similarity(glasses1, glasses2):
    """So sánh đặc trưng kính."""
    if glasses1 is None or glasses2 is None:
        return 0.0
    if glasses1['present'] == glasses2['present']:
        return 1.0
    else:
        return 0.0


def load_known_data(directory):
    """Tải dữ liệu, trích xuất đặc trưng, và ước lượng môi trường."""
    known_face_encodings = []
    known_poses = []
    known_hair_features = []
    known_glasses_features = []
    known_environments = []  # Thêm danh sách môi trường

    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # --- Ước lượng môi trường ---
                time_of_day = get_time_of_day()  # Giả định thời gian chụp là thời gian hiện tại
                lighting = estimate_lighting(image)
                known_environments.append({'time': time_of_day, 'lighting': lighting})

                # --- Tăng cường ảnh dựa trên môi trường ---
                image = enhance_image(image, time_of_day, lighting)

                # --- Xử lý khuôn mặt và các đặc trưng khác ---
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                if len(face_encodings) > 0:
                    for i, face_encoding in enumerate(face_encodings):
                        known_face_encodings.append(face_encoding)
                        hair_features = extract_hair_features(image, face_locations[i])
                        glasses_features = extract_glasses_features(image, face_locations[i])
                        known_hair_features.append(hair_features)
                        known_glasses_features.append(glasses_features)
                else:
                    print(f"Cảnh báo: Không tìm thấy khuôn mặt trong {filename}")

                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3) as pose:
                    results = pose.process(image)
                    if results.pose_landmarks:
                        landmarks_list = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                          for landmark in results.pose_landmarks.landmark]
                        known_poses.append(landmarks_list)
                    else:
                        print(f"Cảnh báo: Không tìm thấy dáng người trong {filename}")
                        known_poses.append(None)

            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")
                continue

    return known_face_encodings, known_poses, known_hair_features, known_glasses_features, known_environments


def calculate_pose_similarity(pose1_landmarks, pose2_landmarks):
    """Tính toán độ tương đồng dáng người."""
    if pose1_landmarks is None or pose2_landmarks is None:
        return 0.0
    important_joints = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    distances = []
    weights = []
    for joint_index in important_joints:
        joint_index = joint_index.value
        x1, y1, _, v1 = pose1_landmarks[joint_index]
        x2, y2, _, v2 = pose2_landmarks[joint_index]
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        distances.append(distance)
        weight = (v1 + v2) / 2.0 * 0.7 + 0.3
        weights.append(weight)
    if not distances:
        return 0.0

    weighted_distances = [d * w for d, w in zip(distances, weights)]
    average_distance = sum(weighted_distances) / sum(weights) if sum(weights) > 0 else 1.0
    similarity = 1.0 / (1.0 + average_distance)
    return similarity


def find_wife(known_face_encodings, known_poses, known_hair_features, known_glasses_features, known_environments):
    """Tìm vợ, có xét đến môi trường."""
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Không thể mở webcam.")
        return

    current_time_of_day = get_time_of_day()  # Thời gian hiện tại

    with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Lỗi khi đọc frame từ webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Ước lượng ánh sáng của frame hiện tại ---
            current_lighting = estimate_lighting(rgb_frame)

            # --- Tăng cường ảnh dựa trên môi trường hiện tại ---
            rgb_frame = enhance_image(rgb_frame, current_time_of_day, current_lighting)

            results = pose.process(rgb_frame)
            current_pose_landmarks = None
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                current_pose_landmarks = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                          for landmark in results.pose_landmarks.landmark]

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            best_pose_similarity = 0.0
            best_hair_similarity = 0.0
            best_glasses_similarity = 0.0
            best_face_distance = float('inf')
            best_environment_similarity = 0.0  # Thêm độ tương đồng môi trường

            if current_pose_landmarks:
                for known_pose in known_poses:
                    if known_pose is not None:
                        pose_similarity = calculate_pose_similarity(known_pose, current_pose_landmarks)
                        if pose_similarity > best_pose_similarity:
                            best_pose_similarity = pose_similarity
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                current_hair = extract_hair_features(rgb_frame, (top, right, bottom, left))
                current_glasses = extract_glasses_features(rgb_frame, (top, right, bottom, left))

                if best_pose_similarity < 0.5:
                    for known_hair, known_glasses, environment in zip(known_hair_features, known_glasses_features,
                                                                      known_environments):
                        hair_similarity = calculate_hair_similarity(known_hair, current_hair)
                        if hair_similarity > best_hair_similarity:
                            best_hair_similarity = hair_similarity

                        glasses_similarity = calculate_glasses_similarity(known_glasses, current_glasses)
                        if glasses_similarity > best_glasses_similarity:
                            best_glasses_similarity = glasses_similarity

                        # So sánh môi trường (rất cơ bản)
                        environment_similarity = 0.0
                        if environment['time'] == current_time_of_day:
                            environment_similarity += 0.5  # Điểm cộng nếu thời gian khớp
                        if environment['lighting'] == current_lighting:
                            environment_similarity += 0.5  # Điểm cộng nếu ánh sáng khớp

                        if environment_similarity > best_environment_similarity:
                            best_environment_similarity = environment_similarity

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    min_face_distance = min(face_distances)
                    if min_face_distance < best_face_distance:
                        best_face_distance = min_face_distance

                # Quyết định
                # Ưu tiên: Dáng -> Môi trường -> Tóc/Kính -> Mặt
                name = "Unknown" # Default name
                color = (0, 0, 255) # Default color is red (Unknown)
                text = f"Unknown"

                if best_pose_similarity > 0.5:
                    color = (0, 255, 0) # Green for "Wife" based on Pose
                    name = "Vo"
                    text = f"Vo (Dang: {best_pose_similarity:.2f}"
                    if best_environment_similarity > 0.7 or best_hair_similarity > 0.6 or best_glasses_similarity > 0.6 or best_face_distance < 0.7:
                        text += ", "
                    if best_environment_similarity > 0.7:
                        text += f"MT: {best_environment_similarity:.2f}, "
                    if best_hair_similarity > 0.6:
                        text += f"Toc: {best_hair_similarity:.2f}, "
                    if best_glasses_similarity > 0.6:
                        text += f"Kinh: {best_glasses_similarity:.2f}, "
                    if best_face_distance < 0.7:
                        text += f"Mat: {best_face_distance:.2f}"
                    text += ")"

                elif best_environment_similarity > 0.7:  # Thêm ưu tiên môi trường
                    color = (0, 255, 255)  # Màu vàng for "Wife" based on Environment
                    name = "Vo"
                    text = f"Vo (MT: {best_environment_similarity:.2f}"
                    if best_hair_similarity > 0.6 or best_glasses_similarity > 0.6 or best_face_distance < 0.7:
                        text += ", "
                    if best_hair_similarity > 0.6:
                        text += f"Toc: {best_hair_similarity:.2f}, "
                    if best_glasses_similarity > 0.6:
                        text += f"Kinh: {best_glasses_similarity:.2f}, "
                    if best_face_distance < 0.7:
                        text += f"Mat: {best_face_distance:.2f}"
                    text += ")"
                elif best_hair_similarity > 0.6 or best_glasses_similarity > 0.6 or best_face_distance < 0.7: # If other features are somewhat matching
                    color = (255, 255, 0) # Cyan for "Possible Vo"
                    name = "Co The La Vo"
                    text = f"Co The La Vo ("
                    if best_hair_similarity > 0.6:
                        text += f"Toc: {best_hair_similarity:.2f}, "
                    if best_glasses_similarity > 0.6:
                        text += f"Kinh: {best_glasses_similarity:.2f}, "
                    if best_face_distance < 0.7:
                        text += f"Mat: {best_face_distance:.2f}"
                    text += ")"


                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.5, color, 1, cv2.LINE_AA)

            cv2.imshow('Webcam - Tim Vo', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    known_faces_dir = "known_faces"  # Thay thế bằng thư mục chứa ảnh khuôn mặt đã biết
    known_face_encodings, known_poses, known_hair_features, known_glasses_features, known_environments = load_known_data(known_faces_dir)
    find_wife(known_face_encodings, known_poses, known_hair_features, known_glasses_features, known_environments)