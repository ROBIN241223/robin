import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import webbrowser
import time
import os
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- Constants ---
HAND_CONFIDENCE_THRESHOLD = 0.7
HAND_TRACKING_THRESHOLD = 0.5
GESTURE_CONFIRM_DURATION = 0.5 # Giữ cử chỉ OK để chụp
CAPTURE_DELAY = 3 # Đếm ngược chụp ảnh (giây)
SAVE_DIRECTORY = "anh_da_luu"
KNOWN_FACES_DIR = "known_faces"
ZALO_PATH = "C:\\Users\\Admin\\AppData\\Local\\Zalo\\Zalo.exe" # SỬA NẾU CẦN
YOUTUBE_URL = "https://www.youtube.com"
FACEBOOK_URL = "https://www.facebook.com"
MIN_TIME_BETWEEN_ACTIONS = 0.7
FACE_TOLERANCE = 0.55
KNOWN_FACE_WIDTH_CM = 14.0
FOCAL_LENGTH = 650 # !! QUAN TRỌNG: CẦN HIỆU CHỈNH CHO WEBCAM CỦA BẠN !!
PROCESS_FACE_EVERY_N_FRAMES = 2

# --- Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=HAND_CONFIDENCE_THRESHOLD,
                        min_tracking_confidence=HAND_TRACKING_THRESHOLD)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam."); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

os.makedirs(SAVE_DIRECTORY, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load Known Faces
print("Đang tải dữ liệu khuôn mặt...")
known_face_encodings = []
known_face_names = []
try:
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(os.path.splitext(filename)[0].replace("_", " "))
                    print(f"- Đã tải: {known_face_names[-1]}")
                else: print(f"Cảnh báo: Không tìm thấy mặt trong {filename}")
            except Exception as e: print(f"Lỗi xử lý {filename}: {e}")
except FileNotFoundError: print(f"Lỗi: Không tìm thấy thư mục '{KNOWN_FACES_DIR}'."); exit()
if not known_face_encodings: print("Cảnh báo: Không có khuôn mặt nào được tải.")

# Volume Control
volume_control_available = False
volume = None
initial_mute_state = False
try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    initial_mute_state = volume.GetMute()
    volume_control_available = True
    print(f"Điều khiển âm lượng sẵn sàng. Mute: {initial_mute_state}")
except Exception as e:
    print(f"Lỗi khởi tạo âm lượng: {e}. Chức năng âm lượng bị vô hiệu hóa.")

# --- State Variables ---
volume_muted = initial_mute_state
last_action_time = 0
capture_count = 0
gesture_start_time = None
capture_ready = False
countdown_start_time = None
frame_count = 0
face_locations, face_encodings, face_names, face_distances_cm = [], [], [], [] # Face processing results

# --- Helper Functions ---
def _check_fingers_up(lm, required_tips, forbidden_tips=None):
    """Kiểm tra ngón tay lên/xuống."""
    if not all(lm[tip].y < lm[tip - 2].y for tip in required_tips): return False
    if forbidden_tips is None:
        all_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
        forbidden_tips = [tip for tip in all_tips if tip not in required_tips]
    return all(lm[tip].y > lm[tip - 2].y for tip in forbidden_tips)

def draw_countdown(frame, remaining_time):
    """Vẽ số đếm ngược."""
    text = str(int(remaining_time) + 1)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 4, 5
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = (frame.shape[1] - w) // 2, (frame.shape[0] + h) // 2
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 5, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

def perform_action(action_func):
    """Thực hiện hành động nếu đủ thời gian chờ."""
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time >= MIN_TIME_BETWEEN_ACTIONS:
        action_func()
        last_action_time = current_time

def open_app(app_name, url=None, path=None):
    """Mở ứng dụng hoặc URL."""
    print(f"Thực hiện: Mở {app_name}...")
    try:
        if url: webbrowser.open(url)
        elif path:
            if os.path.exists(path): os.startfile(path)
            else: print(f"Lỗi: Không tìm thấy đường dẫn {app_name}: {path}"); return
        else: print(f"Lỗi: Thiếu URL/Path cho {app_name}"); return
        print(f"Hoàn thành: Mở {app_name}!")
    except Exception as e: print(f"Lỗi khi mở {app_name}: {e}")

def toggle_mute():
    """Bật/tắt âm lượng."""
    global volume_muted
    if not volume_control_available or volume is None: print("Chức năng âm lượng không khả dụng."); return
    try:
        new_mute_state = not volume_muted
        volume.SetMute(new_mute_state, None)
        volume_muted = new_mute_state
        print(f"Đã {'tắt' if volume_muted else 'bật'} âm lượng!")
    except Exception as e: print(f"Lỗi thay đổi âm lượng: {e}")

def estimate_distance(face_width_pixels):
    """Ước tính khoảng cách đến khuôn mặt."""
    if face_width_pixels <= 0: return -1
    return (KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH) / face_width_pixels

# --- Gesture Definitions (Lambdas) & Actions ---
LM = mp_hands.HandLandmark # Alias for shorter access
gesture_actions = {
    # Key: lambda function checking gesture, Value: function to call
    lambda lm: _check_fingers_up(lm, [LM.INDEX_FINGER_TIP, LM.MIDDLE_FINGER_TIP]):
        lambda: perform_action(lambda: open_app("youtube", url=YOUTUBE_URL)),
    lambda lm: _check_fingers_up(lm, [LM.INDEX_FINGER_TIP, LM.MIDDLE_FINGER_TIP, LM.RING_FINGER_TIP]):
        lambda: perform_action(lambda: open_app("facebook", url=FACEBOOK_URL)),
    lambda lm: _check_fingers_up(lm, [LM.INDEX_FINGER_TIP, LM.MIDDLE_FINGER_TIP, LM.RING_FINGER_TIP, LM.PINKY_TIP]):
        lambda: perform_action(lambda: open_app("zalo", path=ZALO_PATH)),
    # Nắm tay (Tắt/Bật Mute)
    lambda lm: all(lm[tip].y > lm[tip - 2].y for tip in [LM.INDEX_FINGER_TIP, LM.MIDDLE_FINGER_TIP, LM.RING_FINGER_TIP, LM.PINKY_TIP]):
        lambda: perform_action(toggle_mute),
    # Bàn tay mở (Tắt/Bật Mute)
    lambda lm: all(lm[tip].y < lm[tip - 2].y for tip in [LM.INDEX_FINGER_TIP, LM.MIDDLE_FINGER_TIP, LM.RING_FINGER_TIP, LM.PINKY_TIP]) and lm[LM.THUMB_TIP].x < lm[LM.PINKY_MCP].x :
        lambda: perform_action(toggle_mute),
}
# Cử chỉ OK để chụp ảnh (xử lý riêng)
def is_ok_sign(lm):
    dist = np.sqrt((lm[LM.THUMB_TIP].x - lm[LM.INDEX_FINGER_TIP].x)**2 + (lm[LM.THUMB_TIP].y - lm[LM.INDEX_FINGER_TIP].y)**2)
    others_up = _check_fingers_up(lm, [LM.MIDDLE_FINGER_TIP, LM.RING_FINGER_TIP, LM.PINKY_TIP], forbidden_tips=[])
    return dist < 0.07 and others_up

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret: print("Lỗi đọc webcam."); break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    action_performed_this_frame = False
    show_ok_visual = False # Biến cục bộ để hiển thị OK

    # --- Hand Gesture Processing ---
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        landmarks = hand_results.multi_hand_landmarks[0].landmark
        mp_drawing.draw_landmarks(frame, hand_results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Check action gestures
        for gesture_check, action_func in gesture_actions.items():
            if gesture_check(landmarks):
                action_func()
                action_performed_this_frame = True
                break

        # Check OK sign for photo capture
        if not action_performed_this_frame and is_ok_sign(landmarks):
            if gesture_start_time is None:
                gesture_start_time = time.time()
            if time.time() - gesture_start_time >= GESTURE_CONFIRM_DURATION:
                 if not capture_ready and countdown_start_time is None:
                    capture_ready = True
                    countdown_start_time = time.time()
                    print("Bắt đầu đếm ngược chụp ảnh!")
            show_ok_visual = True # Hiển thị OK nếu đang giữ hoặc đang đếm ngược
        elif not capture_ready: # Reset chỉ khi không phải cử chỉ OK và không đang chụp
            gesture_start_time = None

    # --- Face Recognition Processing ---
    if frame_count % PROCESS_FACE_EVERY_N_FRAMES == 0:
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        face_distances_cm = []

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Unknown"
            distance_cm = estimate_distance((right - left) * 2) # Ước tính khoảng cách

            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=FACE_TOLERANCE)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]: name = known_face_names[best_match_index]

            face_names.append(name)
            face_distances_cm.append(distance_cm)
    frame_count += 1

    # --- Drawing Results ---
    # Faces
    for (top, right, bottom, left), name, dist_cm in zip(face_locations, face_names, face_distances_cm):
        top, right, bottom, left = top*2, right*2, bottom*2, left*2 # Scale back
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        label = f"{name}" + (f" ({dist_cm:.1f} cm)" if dist_cm > 0 else "")
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.rectangle(frame, (left, bottom - h - 10), (left + w + 6, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # OK Sign Visual
    if show_ok_visual or capture_ready: # Hiển thị OK nếu đang giữ hoặc đang chụp
         cv2.putText(frame, "OK", (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    # Photo Capture Countdown & Save
    if capture_ready and countdown_start_time is not None:
        remaining = CAPTURE_DELAY - (time.time() - countdown_start_time)
        if remaining > 0:
            draw_countdown(frame, remaining)
        else:
            filename = os.path.join(SAVE_DIRECTORY, f"anh_{time.strftime('%Y%m%d_%H%M%S')}_{capture_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Đã lưu: {filename}")
            capture_count += 1
            capture_ready, countdown_start_time, gesture_start_time = False, None, None # Reset

    # Display
    cv2.imshow('Webcam - Gestures & Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- Cleanup ---
print("Đang thoát...")
cap.release()
cv2.destroyAllWindows()
if 'hands' in locals() and hands: hands.close()
print("Đã thoát.")