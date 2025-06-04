import os
import time

import cv2
import face_recognition  # Sẽ sử dụng dlib backend
import mediapipe as mp
import numpy as np
import pyautogui

# --- Thư viện điều khiển âm lượng ---
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False
    print("--------------------------------------------------------------------")
    print("CẢNH BÁO: Thư viện pycaw không được tìm thấy hoặc không thể import.")
    print("Tính năng điều khiển âm lượng sẽ bị vô hiệu hóa.")
    print("Để bật tính năng này, hãy cài đặt pycaw bằng lệnh:")
    print("pip install pycaw")
    print("--------------------------------------------------------------------")

# --- Kiểm tra và thông báo về hỗ trợ CUDA/GPU ---
# (Giữ nguyên)
print("--- THÔNG TIN HỖ TRỢ GPU/CUDA ---")
DLIB_CUDA_ENABLED = False
try:
    import dlib
    if hasattr(dlib, 'DLIB_USE_CUDA') and dlib.DLIB_USE_CUDA:
        if hasattr(dlib, 'cuda') and dlib.cuda.get_num_devices() > 0:
            DLIB_CUDA_ENABLED = True
            print(f"[INFO] dlib được biên dịch với hỗ trợ CUDA và tìm thấy {dlib.cuda.get_num_devices()} thiết bị CUDA.")
            print("[INFO] Thư viện 'face_recognition' SẼ cố gắng sử dụng CUDA cho các tác vụ nặng.")
        else:
            print("[WARN] dlib được biên dịch với hỗ trợ CUDA nhưng KHÔNG tìm thấy thiết bị CUDA nào.")
            print("[INFO] Thư viện 'face_recognition' sẽ chạy trên CPU.")
    else:
        print("[WARN] dlib KHÔNG được biên dịch với hỗ trợ CUDA.")
        print("[INFO] Thư viện 'face_recognition' sẽ chạy trên CPU.")
except ImportError:
    print("[ERROR] Không tìm thấy thư viện dlib. Cần thiết cho 'face_recognition'.")
print("[INFO] MediaPipe sẽ cố gắng sử dụng GPU (bao gồm CUDA nếu được cấu hình đúng với TensorFlow) nếu có thể.")
print("[INFO] Đảm bảo CUDA Toolkit và cuDNN được cài đặt và cấu hình đúng trong PATH hệ thống của bạn.")
try:
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print(f"[INFO] OpenCL có sẵn trong OpenCV và đã được kích hoạt. Một số hàm cv2 có thể được tăng tốc trên GPU.")
    else:
        print("[INFO] OpenCL không có sẵn hoặc không được kích hoạt trong OpenCV.")
except AttributeError:
    print("[INFO] Phiên bản OpenCV của bạn có thể không có module ocl trực tiếp. T-API vẫn có thể hoạt động ngầm.")
print("---------------------------------")


# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
WINDOW_SCALE_FACTOR = 3 / 5
WINDOW_WIDTH = int(SCREEN_WIDTH * WINDOW_SCALE_FACTOR)
WINDOW_HEIGHT = int(SCREEN_HEIGHT * WINDOW_SCALE_FACTOR)
TOUCHPAD_SENSITIVITY_X = 1.5
TOUCHPAD_SENSITIVITY_Y = 1.5
MIN_VOL_DISTANCE = 30
MAX_VOL_DISTANCE = 250
DRAG_HOLD_DURATION = 0.4
CLICK_DISTANCE_THRESHOLD = 30
PINCH_THRESHOLD = 30 # <-- THÊM LẠI DÒNG NÀY
HAND_OPEN_MIN_FINGERS = 3
KEY_WIDTH = 35
KEY_HEIGHT = 40
KEY_GAP = 5
KEYBOARD_PADDING = 20

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

keyboard_keys = [
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Bksp'],
    ['Tab', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['Caps', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'Enter'],
    ['ShiftL', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'ShiftR'],
    ['CtrlL', 'Space', 'CtrlR']
]
shifted_chars_map = {
    '`': '~', '1': '!', '2': '@', '3': '#', '4': '$', '5': '%', '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
    '-': '_', '=': '+', '[': '{', ']': '}', '\\': '|',
    ';': ':', "'": '"', ',': '<', '.': '>', '/': '?'
}
pyautogui_key_map = {
    'Bksp': 'backspace', 'Tab': 'tab', 'Enter': 'enter', 'Caps': 'capslock',
    'ShiftL': 'shiftleft', 'ShiftR': 'shiftright',
    'CtrlL': 'ctrlleft', 'CtrlR': 'ctrlright',
    'Space': 'space'
}


if PYCAW_AVAILABLE:
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_control = cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as e:
        print(f"Lỗi khi khởi tạo điều khiển âm lượng với pycaw: {e}")
        PYCAW_AVAILABLE = False

def set_system_volume(level_scalar):
    if PYCAW_AVAILABLE and 'volume_control' in globals():
        try:
            volume_control.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level_scalar)), None)
        except Exception as e:
            print(f"Lỗi khi đặt âm lượng: {e}")

def get_system_volume_scalar():
    if PYCAW_AVAILABLE and 'volume_control' in globals():
        try:
            return volume_control.GetMasterVolumeLevelScalar()
        except Exception as e:
            print(f"Lỗi khi lấy âm lượng: {e}")
            return 0.5
    return 0.5

def load_known_faces(directory):
    # (Giữ nguyên)
    known_face_encodings = []
    known_face_names = []
    if not os.path.exists(directory):
        print(f"Thư mục '{directory}' không tồn tại.")
        return known_face_encodings, known_face_names
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                face_encs = face_recognition.face_encodings(image, num_jitters=1)
                if face_encs:
                    known_face_encodings.append(face_encs[0])
                    known_face_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"Cảnh báo: Không tìm thấy khuôn mặt trong '{filename}'.")
            except Exception as e:
                print(f"Lỗi xử lý '{filename}': {e}")
    return known_face_encodings, known_face_names


def draw_keyboard(frame, current_text, hovered_key=None, clicked_key_feedback=None,
                  caps_on=False, shift_on=False, ctrl_on=False, current_mode="keyboard",
                  current_volume_percent=None, is_system_mouse_down_for_drag=False):
    # (Giữ nguyên)
    height, width, _ = frame.shape
    if current_mode == "keyboard":
        max_keys_in_row = max(len(r) for r in keyboard_keys) if keyboard_keys else 0
        keyboard_block_width = max_keys_in_row * (KEY_WIDTH + KEY_GAP) - KEY_GAP if max_keys_in_row > 0 else 0
        if keyboard_block_width > width - KEYBOARD_PADDING * 2:
            keyboard_block_width = width - KEYBOARD_PADDING * 2
        start_x = (width - keyboard_block_width) // 2
        if start_x < KEYBOARD_PADDING: start_x = KEYBOARD_PADDING
        num_rows = len(keyboard_keys)
        keyboard_block_height = num_rows * (KEY_HEIGHT + KEY_GAP) - KEY_GAP
        search_bar_height = 50
        total_keyboard_ui_height = keyboard_block_height + search_bar_height + 10
        start_y = height - total_keyboard_ui_height - 20
        if start_y < 20: start_y = 20
        search_bar_y_start = start_y - search_bar_height - 10
        search_bar_actual_width = keyboard_block_width
        if start_x + search_bar_actual_width > width - KEYBOARD_PADDING:
            search_bar_actual_width = width - KEYBOARD_PADDING - start_x
        cv2.rectangle(frame, (start_x, search_bar_y_start),
                      (start_x + search_bar_actual_width, search_bar_y_start + search_bar_height),
                      (220, 220, 220), -1)
        cv2.rectangle(frame, (start_x, search_bar_y_start),
                      (start_x + search_bar_actual_width, search_bar_y_start + search_bar_height),
                      (0, 0, 0), 1)
        font_scale_search = 0.8;
        thickness_search = 2
        max_text_width_search = search_bar_actual_width - 20
        display_text_search = current_text
        (text_w_search, _), _ = cv2.getTextSize(display_text_search, cv2.FONT_HERSHEY_SIMPLEX, font_scale_search,
                                                thickness_search)
        while text_w_search > max_text_width_search and len(display_text_search) > 0:
            display_text_search = display_text_search[1:]
            (text_w_search, _), _ = cv2.getTextSize(display_text_search, cv2.FONT_HERSHEY_SIMPLEX, font_scale_search,
                                                    thickness_search)
        if not display_text_search and current_text: display_text_search = "..."
        text_y_search = search_bar_y_start + (search_bar_height +
                                              cv2.getTextSize(display_text_search, cv2.FONT_HERSHEY_SIMPLEX,
                                                              font_scale_search, thickness_search)[0][1]) // 2
        cv2.putText(frame, display_text_search, (start_x + 10, text_y_search),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_search, (0, 0, 0), thickness_search)
        keyboard_content_start_y = start_y
        for i, row_keys_list in enumerate(keyboard_keys):
            row_content_width = len(row_keys_list) * (KEY_WIDTH + KEY_GAP) - KEY_GAP
            current_row_start_x = start_x + (keyboard_block_width - row_content_width) // 2
            if current_row_start_x < KEYBOARD_PADDING: current_row_start_x = KEYBOARD_PADDING
            for j, key_char in enumerate(row_keys_list):
                x_pos = current_row_start_x + j * (KEY_WIDTH + KEY_GAP)
                y_pos = keyboard_content_start_y + i * (KEY_HEIGHT + KEY_GAP)
                if x_pos + KEY_WIDTH > width - KEYBOARD_PADDING: continue
                if y_pos + KEY_HEIGHT > height - 20: continue
                key_bg_color = (255, 255, 255);
                key_text_color = (0, 0, 0)
                if key_char == 'Caps' and caps_on:
                    key_bg_color = (100, 180, 100)
                elif key_char in ['ShiftL', 'ShiftR'] and shift_on:
                    key_bg_color = (100, 180, 100)
                elif key_char in ['CtrlL', 'CtrlR'] and ctrl_on:
                    key_bg_color = (100, 180, 100)
                if key_char == clicked_key_feedback:
                    key_bg_color = (0, 255, 0)
                elif key_char == hovered_key and key_char != clicked_key_feedback:
                    is_modifier_active = (key_char == 'Caps' and caps_on) or \
                                         (key_char in ['ShiftL', 'ShiftR'] and shift_on) or \
                                         (key_char in ['CtrlL', 'CtrlR'] and ctrl_on)
                    key_bg_color = (120, 200, 120) if is_modifier_active else (180, 220, 180)
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + KEY_WIDTH, y_pos + KEY_HEIGHT), key_bg_color, -1)
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + KEY_WIDTH, y_pos + KEY_HEIGHT), (0, 0, 0), 1)
                font_scale = 0.55 if len(key_char) > 1 and key_char != "Space" else 0.65
                if key_char == "Space": font_scale = 0.45
                text_thickness = 1
                (text_w_key, text_h_key), _ = cv2.getTextSize(key_char, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              text_thickness)
                text_x_key = x_pos + (KEY_WIDTH - text_w_key) // 2
                text_y_key = y_pos + (KEY_HEIGHT + text_h_key) // 2
                cv2.putText(frame, key_char, (text_x_key, text_y_key), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            key_text_color, text_thickness)

    mode_text = f"Mode: {current_mode.upper()}"
    if current_mode == "touchpad" and is_system_mouse_down_for_drag:
        mode_text += " (Dragging)"
    cv2.putText(frame, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    if PYCAW_AVAILABLE and current_volume_percent is not None:
        vol_display_text = f"Volume: {current_volume_percent:.0f}%"
        cv2.putText(frame, vol_display_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

# --- START: New Gesture Functions ---
def get_distance(p1, p2):
    """Calculates 2D distance between two landmarks."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2) * 1000 # Scale up

def is_finger_extended(hand_landmarks, tip_idx, mcp_idx):
    """Checks if a finger is extended based on distance from wrist."""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    tip = hand_landmarks.landmark[tip_idx]
    mcp = hand_landmarks.landmark[mcp_idx]
    # Finger is up if tip is further from wrist than MCP (add a buffer)
    return get_distance(tip, wrist) > get_distance(mcp, wrist) * 1.1

def is_finger_open(hand_landmarks, mcp_idx, pip_idx, tip_idx):
    """Checks if a finger is generally straight/open (Y-based)."""
    mcp_y = hand_landmarks.landmark[mcp_idx].y
    pip_y = hand_landmarks.landmark[pip_idx].y
    tip_y = hand_landmarks.landmark[tip_idx].y
    return tip_y < pip_y and pip_y < mcp_y

def get_hand_gesture_new(hand_landmarks):
    """Determines gesture based on new distance rules."""
    try:
        index_up = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)
        middle_up = is_finger_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)

        # Check for Volume: Index DOWN, Middle UP
        if not index_up and middle_up:
            return "volume"

        # Check for Touchpad: Index UP, Middle DOWN
        if index_up and not middle_up:
            return "touchpad"

        # Check for Keyboard (Open Hand)
        index_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_TIP)
        middle_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
        ring_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_TIP)
        pinky_open = is_finger_open(hand_landmarks, mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_TIP)
        if sum([index_open, middle_open, ring_open, pinky_open]) >= HAND_OPEN_MIN_FINGERS:
             return "keyboard"

    except Exception:
        return None # Return None on error
    return "touchpad" # Default to touchpad if no specific gesture matches but hand is seen

def is_clicking_gesture_new(hand_landmarks, threshold=CLICK_DISTANCE_THRESHOLD):
    """Checks if Thumb Tip (4) is close to Index MCP (5)."""
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        dist = get_distance(thumb_tip, index_mcp)
        return dist < threshold
    except Exception:
        return False

def is_pinching_gesture(hand_landmarks, threshold=PINCH_THRESHOLD):
    """Checks for normal pinch (Thumb Tip 4 to Index Tip 8)."""
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        dist = get_distance(thumb_tip, index_tip)
        return dist < threshold
    except Exception:
        return False
# --- END: New Gesture Functions ---


def get_char_for_typing(key_char, shift_active, caps_active):
    # (Giữ nguyên)
    if key_char.isalpha() and len(key_char) == 1:
        is_upper = (caps_active and not shift_active) or (not caps_active and shift_active)
        return key_char.upper() if is_upper else key_char.lower()
    elif shift_active and key_char in shifted_chars_map:
        return shifted_chars_map[key_char]
    elif not shift_active and key_char in shifted_chars_map:
        return key_char
    elif key_char.isdigit():
        return key_char
    return None

# ... (Giữ nguyên các phần code từ đầu đến trước hàm show_camera_interface) ...

def show_camera_interface():
    WINDOW_NAME = "Virtual Keyboard/Touchpad - Gesture Control"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Lỗi: Không thể mở camera."); cv2.destroyWindow(WINDOW_NAME); return

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) # Chỉ xử lý 1 tay
    current_text = ""
    last_keyboard_click_time = 0;
    click_cooldown = 0.5
    hovered_key_char = None;
    clicked_key_feedback = None
    clicked_key_feedback_start_time = 0;
    click_feedback_duration = 0.2
    is_caps_lock_on = False;
    is_shift_on = False;
    is_ctrl_on = False
    current_mode = "keyboard";
    current_volume_for_display = None

    is_dragging = False
    click_gesture_start_time = 0.0
    last_click_state = False

    print("Giao diện camera đang hoạt động.")
    print("- Mở bàn tay => Chế độ Bàn phím (Keyboard).")
    print("- Duỗi ngón TRỎ, Gập ngón GIỮA => Chế độ Chuột cảm ứng (Touchpad).")
    print("- Gập ngón TRỎ, Duỗi ngón GIỮA => Chế độ Âm lượng (Volume).")
    print("---")
    print("- Bàn phím: Chụm ngón TRỎ & CÁI để nhấn.")
    print("- Touchpad: Di chuột bằng NGÓN TRỎ.")
    print("           Chạm ngón CÁI vào GỐC ngón trỏ để CLICK.")
    print("           Chạm & GIỮ ngón CÁI vào GỐC ngón trỏ để KÉO THẢ (VẪN DI CHUYỂN ĐƯỢC).") # Cập nhật hướng dẫn
    print("- Volume: Thay đổi KHOẢNG CÁCH giữa ngón CÁI & GIỮA để chỉnh.")
    print("---")
    print("- Nhấn 'q' để thoát, 'c' để xóa text (chế độ bàn phím).")

    while True:
        ret, frame_original = cap.read()
        if not ret: print("Lỗi đọc frame."); break

        frame = cv2.resize(frame_original, (WINDOW_WIDTH, WINDOW_HEIGHT))
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        current_loop_time = time.time()
        if clicked_key_feedback and (current_loop_time - clicked_key_feedback_start_time > click_feedback_duration):
            clicked_key_feedback = None

        thumb_tip_coord = None
        index_tip_coord = None
        middle_tip_coord = None
        current_gesture = None
        is_clicking_now = False
        is_pinching_now = False


        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                   mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

            thumb_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_lm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            thumb_tip_coord = (int(thumb_lm.x * w_frame), int(thumb_lm.y * h_frame))
            index_tip_coord = (int(index_lm.x * w_frame), int(index_lm.y * h_frame))
            middle_tip_coord = (int(middle_lm.x * w_frame), int(middle_lm.y * h_frame))

            current_gesture = get_hand_gesture_new(hand_landmarks)
            is_clicking_now = is_clicking_gesture_new(hand_landmarks)
            is_pinching_now = is_pinching_gesture(hand_landmarks)

            if current_gesture:
                 cv2.putText(frame, f"Gesture: {current_gesture.upper()}",
                            (w_frame - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if is_clicking_now:
                 cv2.circle(frame, thumb_tip_coord, 10, (0, 0, 255), -1)
                 cv2.putText(frame, "CLICKING!", (w_frame - 250, 90),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


        # Determine current mode
        old_mode = current_mode
        if current_gesture == "volume": current_mode = "volume"
        elif current_gesture == "touchpad": current_mode = "touchpad"
        elif current_gesture == "keyboard": current_mode = "keyboard"
        else:
            if not results.multi_hand_landmarks: current_mode = "keyboard"

        if old_mode != current_mode:
            print(f"Chuyển sang chế độ: {current_mode.upper()}")
            if is_dragging:
                pyautogui.mouseUp(); is_dragging = False; click_gesture_start_time = 0.0
            current_text = ""; hovered_key_char = None

        # --- Execute actions based on mode ---
        current_volume_for_display = None
        hovered_key_char = None

        if current_mode == "volume":
            if PYCAW_AVAILABLE and thumb_tip_coord and middle_tip_coord:
                tx, ty = thumb_tip_coord
                mx, my = middle_tip_coord
                cv2.line(frame, (tx, ty), (mx, my), (255, 255, 0), 2)
                distance = np.sqrt((tx - mx)**2 + (ty - my)**2)
                vol_scalar = (distance - MIN_VOL_DISTANCE) / (MAX_VOL_DISTANCE - MIN_VOL_DISTANCE)
                vol_scalar = max(0.0, min(1.0, vol_scalar))
                set_system_volume(vol_scalar)
                current_volume_for_display = vol_scalar * 100

        elif current_mode == "touchpad":
            if index_tip_coord:
                # 1. Di chuyển chuột bằng ngón trỏ
                ix, iy = index_tip_coord
                move_x_norm = ix / w_frame
                move_y_norm = iy / h_frame
                mouse_x = int(move_x_norm * SCREEN_WIDTH * TOUCHPAD_SENSITIVITY_X - (SCREEN_WIDTH * (TOUCHPAD_SENSITIVITY_X - 1) / 2))
                mouse_y = int(move_y_norm * SCREEN_HEIGHT * TOUCHPAD_SENSITIVITY_Y - (SCREEN_HEIGHT * (TOUCHPAD_SENSITIVITY_Y - 1) / 2))
                mouse_x = max(0, min(mouse_x, SCREEN_WIDTH - 1))
                mouse_y = max(0, min(mouse_y, SCREEN_HEIGHT - 1))

                # LUÔN LUÔN DI CHUYỂN CHUỘT THEO NGÓN TRỎ KHI Ở CHẾ ĐỘ TOUCHPAD
                pyautogui.moveTo(mouse_x, mouse_y, duration=0)
                # ==========================================================

                cv2.circle(frame, index_tip_coord, 8, (255, 0, 255), -1)

                # 2. Xử lý Click/Drag (4 chạm 5)
                if is_clicking_now:
                    if not is_dragging and not last_click_state:
                        click_gesture_start_time = current_loop_time
                        print("Click State Entered")
                    elif not is_dragging and (current_loop_time - click_gesture_start_time > DRAG_HOLD_DURATION) and click_gesture_start_time > 0:
                        print("Touchpad: Mouse Down (Drag Start - 4to5)")
                        pyautogui.mouseDown()
                        is_dragging = True
                elif not is_clicking_now and last_click_state:
                     if is_dragging:
                         print("Touchpad: Mouse Up (Drag End - 4to5 release)")
                         pyautogui.mouseUp()
                         is_dragging = False
                     elif click_gesture_start_time > 0.0:
                         print("Touchpad: Quick Click (4to5 touch-release)")
                         pyautogui.click()
                     click_gesture_start_time = 0.0

        elif current_mode == "keyboard":
             if index_tip_coord:
                ix_k, iy_k = index_tip_coord
                # Click (Dùng Pinch 4-8)
                if is_pinching_now and (current_loop_time - last_keyboard_click_time > click_cooldown):
                    # ... (Logic click bàn phím bằng Pinch như cũ) ...
                    max_keys_kb = max(len(r) for r in keyboard_keys) if keyboard_keys else 0
                    kb_block_width_calc = max_keys_kb * (KEY_WIDTH + KEY_GAP) - KEY_GAP if max_keys_kb > 0 else 0
                    drawable_kb_width = w_frame - KEYBOARD_PADDING * 2
                    if kb_block_width_calc > drawable_kb_width: kb_block_width_calc = drawable_kb_width
                    start_x_kb_base = (w_frame - kb_block_width_calc) // 2
                    if start_x_kb_base < KEYBOARD_PADDING: start_x_kb_base = KEYBOARD_PADDING
                    num_rows_kb = len(keyboard_keys)
                    kb_block_height_calc = num_rows_kb * (KEY_HEIGHT + KEY_GAP) - KEY_GAP
                    search_bar_h_calc = 50; total_kb_ui_h_calc = kb_block_height_calc + search_bar_h_calc + 10
                    start_y_kb = h_frame - total_kb_ui_h_calc - 20
                    if start_y_kb < 20: start_y_kb = 20
                    key_clicked_this_loop = False
                    for i_row, row_layout in enumerate(keyboard_keys):
                        if key_clicked_this_loop: break
                        row_w_current = len(row_layout) * (KEY_WIDTH + KEY_GAP) - KEY_GAP
                        current_row_start_x_keys = start_x_kb_base + (kb_block_width_calc - row_w_current) // 2
                        if current_row_start_x_keys < KEYBOARD_PADDING: current_row_start_x_keys = KEYBOARD_PADDING
                        for j_col, key_char_value in enumerate(row_layout):
                            key_rect_x = current_row_start_x_keys + j_col * (KEY_WIDTH + KEY_GAP)
                            key_rect_y = start_y_kb + i_row * (KEY_HEIGHT + KEY_GAP)
                            if key_rect_x + KEY_WIDTH > w_frame - KEYBOARD_PADDING: continue
                            if key_rect_y + KEY_HEIGHT > h_frame - 20: continue
                            if key_rect_x < ix_k < key_rect_x + KEY_WIDTH and \
                               key_rect_y < iy_k < key_rect_y + KEY_HEIGHT:
                                print(f"Keyboard Clicked (Pinch): '{key_char_value}'")
                                clicked_key_feedback = key_char_value
                                clicked_key_feedback_start_time = current_loop_time
                                last_keyboard_click_time = current_loop_time
                                # (Xử lý nhấn phím như cũ)
                                if key_char_value == 'Caps': is_caps_lock_on = not is_caps_lock_on
                                elif key_char_value in ['ShiftL', 'ShiftR']: is_shift_on = not is_shift_on
                                elif key_char_value in ['CtrlL', 'CtrlR']: is_ctrl_on = not is_ctrl_on
                                pyautogui_key = pyautogui_key_map.get(key_char_value)
                                if pyautogui_key: pyautogui.press(pyautogui_key)
                                else:
                                    char_to_type = get_char_for_typing(key_char_value, is_shift_on, is_caps_lock_on)
                                    if char_to_type:
                                        pyautogui.typewrite(char_to_type)
                                        current_text += char_to_type
                                    if is_shift_on and key_char_value not in ['ShiftL', 'ShiftR']: is_shift_on = False
                                if key_char_value == 'Bksp': current_text = current_text[:-1] if current_text else ""
                                elif key_char_value == 'Space' and not pyautogui_key: current_text += " "
                                elif key_char_value == 'Tab' and not pyautogui_key: current_text += "    "
                                key_clicked_this_loop = True; break
                        if key_clicked_this_loop: break
                # Hover (Dùng ngón trỏ)
                elif not clicked_key_feedback:
                    # ... (Logic hover như cũ) ...
                    max_keys_kb_hover = max(len(r) for r in keyboard_keys) if keyboard_keys else 0
                    kb_block_width_hover = max_keys_kb_hover * (KEY_WIDTH + KEY_GAP) - KEY_GAP if max_keys_kb_hover > 0 else 0
                    drawable_kb_width_hover = w_frame - KEYBOARD_PADDING * 2
                    if kb_block_width_hover > drawable_kb_width_hover: kb_block_width_hover = drawable_kb_width_hover
                    start_x_kb_base_hover = (w_frame - kb_block_width_hover) // 2
                    if start_x_kb_base_hover < KEYBOARD_PADDING: start_x_kb_base_hover = KEYBOARD_PADDING
                    num_rows_kb_hover = len(keyboard_keys); kb_block_height_hover = num_rows_kb_hover * (KEY_HEIGHT + KEY_GAP) - KEY_GAP
                    search_bar_h_hover = 50; total_kb_ui_h_hover = kb_block_height_hover + search_bar_h_hover + 10
                    start_y_kb_hover = h_frame - total_kb_ui_h_hover - 20
                    if start_y_kb_hover < 20: start_y_kb_hover = 20
                    hover_found_this_loop = False
                    for i_r, r_layout in enumerate(keyboard_keys):
                        if hover_found_this_loop: break
                        r_w_current = len(r_layout) * (KEY_WIDTH + KEY_GAP) - KEY_GAP
                        current_r_start_x = start_x_kb_base_hover + (kb_block_width_hover - r_w_current) // 2
                        if current_r_start_x < KEYBOARD_PADDING: current_r_start_x = KEYBOARD_PADDING
                        for j_c, k_char_val in enumerate(r_layout):
                            k_rect_x = current_r_start_x + j_c * (KEY_WIDTH + KEY_GAP)
                            k_rect_y = start_y_kb_hover + i_r * (KEY_HEIGHT + KEY_GAP)
                            if k_rect_x < ix_k < k_rect_x + KEY_WIDTH and \
                               k_rect_y < iy_k < k_rect_y + KEY_HEIGHT:
                                hovered_key_char = k_char_val
                                hover_found_this_loop = True; break
                        if hover_found_this_loop: break

        last_click_state = is_clicking_now

        draw_keyboard(frame, current_text, hovered_key_char, clicked_key_feedback,
                      is_caps_lock_on, is_shift_on, is_ctrl_on, current_mode,
                      current_volume_for_display, is_dragging)
        cv2.imshow(WINDOW_NAME, frame)

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'): print("Thoát..."); break
        elif key_press == ord('c') and current_mode == "keyboard": current_text = ""; print("Đã xóa văn bản.")

    cap.release(); cv2.destroyAllWindows()
    if 'hands' in locals() and hands: hands.close()

# ... (Giữ nguyên các hàm recognize_faces_and_launch_interface và __main__) ...

def recognize_faces_and_launch_interface(known_face_encodings, known_face_names, target_face_name="DUC DEP TRAI"):
    WINDOW_NAME_FACE_REC = "Face Recognition - Press Q to Quit"
    cv2.namedWindow(WINDOW_NAME_FACE_REC, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME_FACE_REC, WINDOW_WIDTH, WINDOW_HEIGHT)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Lỗi: Không thể mở webcam cho nhận diện khuôn mặt.")
        cv2.destroyWindow(WINDOW_NAME_FACE_REC); return

    print("Đang nhận diện khuôn mặt... Tìm kiếm: ", target_face_name)
    face_recognized_and_launched = False
    detection_model = "cnn" if DLIB_CUDA_ENABLED else "hog"
    if DLIB_CUDA_ENABLED:
        print(f"[INFO] Sử dụng model '{detection_model}' cho face_locations (tăng tốc GPU).")
    else:
        print(f"[INFO] Sử dụng model '{detection_model}' cho face_locations (CPU).")

    while not face_recognized_and_launched:
        ret, frame_original_face = video_capture.read()
        if not ret: print("Lỗi khi đọc frame từ webcam."); break
        frame_face = cv2.resize(frame_original_face, (WINDOW_WIDTH, WINDOW_HEIGHT))
        frame_face = cv2.flip(frame_face, 1)
        rgb_frame = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
        face_locations_in_frame = face_recognition.face_locations(rgb_frame, model=detection_model)
        face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations_in_frame, num_jitters=1)

        for face_encoding, face_location in zip(face_encodings_in_frame, face_locations_in_frame):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            top, right, bottom, left = face_location
            cv2.rectangle(frame_face, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame_face, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            if name == target_face_name:
                print(f"Gương mặt đã nhận diện: {name}. Khởi động giao diện điều khiển.")
                video_capture.release(); cv2.destroyWindow(WINDOW_NAME_FACE_REC)
                show_camera_interface(); face_recognized_and_launched = True; break
        if face_recognized_and_launched: break
        cv2.imshow(WINDOW_NAME_FACE_REC, frame_face)
        if cv2.waitKey(1) & 0xFF == ord('q'): print("Thoát khỏi nhận diện khuôn mặt."); break
    if video_capture.isOpened(): video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    known_faces_dir = "known_faces";
    target_user_name = "DUC DEP TRAI"
    if not os.path.exists(known_faces_dir):
        try:
            os.makedirs(known_faces_dir)
            print(f"Đã tạo thư mục '{known_faces_dir}'.")
            print(f"Vui lòng thêm ảnh khuôn mặt (ví dụ: {target_user_name}.jpg) vào thư mục này.")
            exit()
        except OSError as e:
            print(f"Lỗi khi tạo thư mục '{known_faces_dir}': {e}"); exit()
    print(f"Đang tải các khuôn mặt đã biết từ thư mục: '{known_faces_dir}'...")
    known_encodings, known_names = load_known_faces(known_faces_dir)
    if not known_encodings:
        print(f"Không tìm thấy khuôn mặt hợp lệ nào trong thư mục '{known_faces_dir}'.")
        print(f"Vui lòng thêm ảnh (ví dụ: {target_user_name}.jpg) và chạy lại chương trình.")
    elif target_user_name not in known_names:
        print(f"Không tìm thấy khuôn mặt của '{target_user_name}' trong các khuôn mặt đã biết.")
        print(f"Các khuôn mặt đã biết: {', '.join(known_names) if known_names else 'Không có'}")
        print(f"Vui lòng đảm bảo có file ảnh '{target_user_name}.jpg' (hoặc .png/.jpeg) trong thư mục '{known_faces_dir}'.")
    else:
        print(f"Đã tải {len(known_names)} khuôn mặt đã biết: {', '.join(known_names)}")
        recognize_faces_and_launch_interface(known_encodings, known_names, target_face_name=target_user_name)
    print("Chương trình kết thúc")